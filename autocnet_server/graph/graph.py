from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
import os
import pickle
import sys
import time
import threading

import sqlalchemy

from geoalchemy2.shape import to_shape, from_shape
from geoalchemy2.elements import WKTElement, WKBElement
from shapely.geometry import Point
import shapely

import csmapi
import networkx as nx
from networkx.classes.reportviews import NodeView, EdgeView
import numpy as np
import pandas as pd
from plio.utils import utils as io_utils
import pyproj

from redis import StrictRedis

import yaml

#Load the config file
with open(os.environ['autocnet_config'], 'r') as f:
    config = yaml.load(f)

# Patch in dev. versions if requested.
acp = config.get('developer', {}).get('autocnet_path', None)
if acp:
    sys.path.insert(0, acp)

asp = config.get('developer', {}).get('autocnet_server_path', None)
if asp:
    sys.path.insert(0, asp)

from autocnet.graph import node, network, edge
from autocnet.graph.node import Node
from autocnet.utils import utils
from autocnet.io import keypoints as io_keypoints
from autocnet.transformation.fundamental_matrix import compute_reprojection_error

from plio.utils import generate_vrt

from plurmy import spawn, spawn_jobarr, slurm_walltime_to_seconds
from autocnet_server import Session, engine
from autocnet_server.db.model import Images, Keypoints, Matches, Cameras, Network, Base, Overlay, Edges, Costs
from autocnet_server.db.connection import new_connection, Parent
from autocnet_server.db.wrappers import DbDataFrame
from autocnet_server.sensors.csm import create_camera, generate_latlon_footprint


class NetworkNode(Node):
    def __init__(self, *args, parent=None, **kwargs):
        # If this is the first time that the image is seen, add it to the DB
        if parent is None:
            self.parent = Parent(config)
        else:
            self.parent = parent
        
        # Create a session to work in
        session = Session()
        # For now, just use the PATH to determine if the node/image is in the DB
        res = session.query(Images).filter(Images.path == kwargs['image_path']).first()
        exists = False
        if res:
            exists = True
            kwargs['node_id'] = res.id
        session.close()
        super(NetworkNode, self).__init__(*args, **kwargs)
        
        if exists is False:
            # Create the camera entry
            try:
                self._camera = create_camera(self.geodata)
                serialized_camera = self._camera.getModelState()
                cam = Cameras(camera=serialized_camera)
            except:
                cam = None
            kpspath = io_keypoints.create_output_path(self.geodata)

            # Create the keypoints entry
            kps = Keypoints(path=kpspath, nkeypoints=0)
            # Create the image
            i = Images(name=kwargs['image_name'],
                       path=kwargs['image_path'],
                       footprint_latlon=self.footprint,
                       cameras=cam, keypoints=kps)
            session = Session()
            session.add(i)
            session.commit()
            session.close()
        self.job_status = defaultdict(dict)

    def _from_db(self, table_obj, key='image_id'):
        """
        Generic database query to pull the row associated with this node
        from an arbitrary table. We assume that the row id matches the node_id.

        Parameters
        ----------
        table_obj : object
                    The declared table class (from db.model)

        key : str
              The name of the column to compare this object's node_id with. For
              most tables this will be the default, 'image_id' because 'image_id'
              is the foreign key in the DB. For the Images table (the parent table),
              the key is simply 'id'.
        """
        if 'node_id' not in self.keys():
            return
        session = Session()
        res = session.query(table_obj).filter(getattr(table_obj,key) == self['node_id']).first()
        session.close()
        return res

    @property
    def keypoint_file(self):
        res = self._from_db(Keypoints)
        if res is None:
            return 
        return res.path

    @property
    def keypoints(self):
        try:
            return io_keypoints.from_hdf(self.keypoint_file, descriptors=False)
        except:
            return pd.DataFrame()
            
    @keypoints.setter
    def keypoints(self, kps):
        session = Session()
        io_keypoints.to_hdf(self.keypoint_file, keypoints=kps)
        res = session.query(Keypoints).filter(getattr(Keypoints,'image_id') == self['node_id']).first()

        if res is None:
            _ = self.keypoint_file
            res = self._from_db(Keypoints)
        res.nkeypoints = len(kps)
        session.commit()

    @property
    def descriptors(self):
        try:
            return io_keypoints.from_hdf(self.keypoint_file, keypoints=False)
        except:
            return

    @descriptors.setter
    def descriptors(self, desc):
        io_keypoints.to_hdf(self.keypoint_file, descriptors=desc)

    @property
    def nkeypoints(self):
        """
        Get the number of keypoints from the database
        """
        res = self._from_db(Keypoints)
        nkps = res.nkeypoints
        return nkps

    @property
    def camera(self):
        """
        Get the camera object from the database.
        """
        if not getattr(self, '_camera', None):
            res = self._from_db(Cameras)
            if res is not None:
                plugin = csmapi.Plugin.findPlugin('USGS_ASTRO_LINE_SCANNER_PLUGIN')
                self._camera = plugin.constructModelFromState(res.camera)
        return self._camera
    
    @property
    def footprint(self):
        
        res = Session().query(Images).filter(Images.id == self['node_id']).first()
        if res is None:
            try:
                footprint_latlon = generate_latlon_footprint(self.camera)
                footprint_latlon = footprint_latlon.ExportToWkt()
                footprint_latlon = WKTElement(footprint_latlon, srid=config['spatial']['srid'])
            except:
                footprint_latlon = None
        else:
            footprint_latlon = to_shape(res.footprint_latlon)
        return footprint_latlon

    def generate_vrt(self, **kwargs):
        """
        Using the image footprint, generate a VRT to that is usable inside
        of a GIS (QGIS or ArcGIS) to visualize a warped version of this image.
        """
        outpath = config['directories']['vrt_dir']
        generate_vrt.warped_vrt(self.camera, self.geodata.raster_size,
                                self.geodata.file_name, outpath=outpath)


class NetworkEdge(edge.Edge):

    default_msg = {'sidx':None,
                    'didx':None,
                    'task':None,
                    'param_step':0,
                    'success':False}

    def __init__(self, *args, **kwargs):
        super(NetworkEdge, self).__init__(*args, **kwargs)
        self.job_status = defaultdict(dict)

    def _from_db(self, table_obj):
        """
        Generic database query to pull the row associated with this node
        from an arbitrary table. We assume that the row id matches the node_id.

        Parameters
        ----------
        table_obj : object
                    The declared table class (from db.model)
        """
        session = Session()
        res = session.query(table_obj).\
               filter(table_obj.source == self.source['node_id']).\
               filter(table_obj.destination == self.destination['node_id'])
        session.close()
        return res

    @property 
    def masks(self):
        res = Session().query(Edges.masks).\
                                        filter(Edges.source == self.source['node_id']).\
                                        filter(Edges.destination == self.destination['node_id']).\
                                        first()
        
        if res:
            df = pd.DataFrame.from_records(res[0])
            df.index = df.index.map(int)
        else:
            ids = list(map(int, self.matches.index.values))
            df = pd.DataFrame(index=ids)
        df.index.name = 'match_id'
        return DbDataFrame(df, parent=self, name='masks')

    @masks.setter
    def masks(self, v):

        def dict_check(input):
            for k, v in input.items():
                if isinstance(v, dict):
                    dict_check(v)
                elif v is None:
                    continue
                elif np.isnan(v):
                    input[k] = None
            

        df = pd.DataFrame(v)
        session = Session()
        res = session.query(Edges).\
                                filter(Edges.source == self.source['node_id']).\
                                filter(Edges.destination == self.destination['node_id']).first()
        if res:
            as_dict = df.to_dict()
            dict_check(as_dict)
            # Update the masks
            res.masks = as_dict
            session.add(res)
            session.commit()

    @property
    def costs(self):
        # these are np.float coming out, sqlalchemy needs ints
        ids = list(map(int, self.matches.index.values))
        res = Session().query(Costs).filter(Costs.match_id.in_(ids)).all()
        #qf = q.filter(Costs.match_id.in_(ids))
        
        if res:
        # Parse the JSON dicts in the cost field into a full dimension dataframe
            costs = {r.match_id:r._cost for r in res}
            df = pd.DataFrame.from_records(costs).T  # From records is important because from_dict drops rows with empty dicts
        else:
            df = pd.DataFrame(index=ids)

        df.index.name = 'match_id'
        return DbDataFrame(df, parent=self, name='costs')


    @costs.setter
    def costs(self, v):
        to_db_add = []
        # Get the query obj
        session = Session()
        q = session.query(Costs)
        # Need the new instance here to avoid __setattr__ issues
        df = pd.DataFrame(v)
        for idx, row in df.iterrows():
            # Now invert the expanded dict back into a single JSONB column for storage
            res = q.filter(Costs.match_id == idx).first()
            if res:
                #update the JSON blob
                costs_new_or_updated = row.to_dict()
                for k, v in costs_new_or_updated.items():
                    if v is None:
                        continue
                    elif np.isnan(v):
                        v = None
                    res._cost[k] = v
                sqlalchemy.orm.attributes.flag_modified(res, '_cost')
                session.add(res)
                session.commit()
            else:
                row = row.to_dict()
                costs = row.pop('_costs', {})
                for k, v in row.items():
                    if np.isnan(v):
                        v = None
                    costs[k] = v
                cost = Costs(match_id=idx, _cost=costs)
                to_db_add.append(cost)
        if to_db_add:
            session.bulk_save_objects(to_db_add)
        session.commit()

    @property
    def matches(self):
        session = Session()
        q = session.query(Matches)
        qf = q.filter(Matches.source == self.source['node_id'],
                      Matches.destination == self.destination['node_id'])
        odf = pd.read_sql(qf.statement, q.session.bind).set_index('id')
        df = pd.DataFrame(odf.values, index=odf.index.values, columns=odf.columns.values)
        df.index.name = 'id'
        # Explicit close to get the session cleaned up
        session.close()
        return DbDataFrame(df, 
                           parent=self,
                           name='matches')

    @matches.setter
    def matches(self, v):
        to_db_add = []
        to_db_update = []
        df = pd.DataFrame(v)
        df.index.name = v.index.name
        # Get the query obj
        session = Session()
        q = session.query(Matches)
        for idx, row in df.iterrows():
            # Determine if this is an update or the addition of a new row
            if hasattr(row, 'id'):
                res = q.filter(Matches.id == row.id).first()
                match_id = row.id
            elif v.index.name == 'id':
                res = q.filter(Matches.id == row.name).first()
                match_id = row.name
            else:
                res = None
            if res:
                # update
                mapping = {}
                mapping['id'] = match_id
                for index in row.index:
                    row_val = row[index]
                    if isinstance(row_val, (np.int,)):
                        row_val = int(row_val)
                    elif isinstance(row_val, (np.float,)):
                        row_val = float(row_val)
                    elif isinstance(row_val, WKBElement):
                        continue
                    mapping[index] = row_val
                to_db_update.append(mapping)
            else:
                match = Matches(source=int(row.source), source_idx=int(row.source_idx),
                            destination=int(row.destination), destination_idx=int(row.destination_idx))
                to_db_add.append(match)
        if to_db_add:
            session.bulk_save_objects(to_db_add)
        if to_db_update:
            session.bulk_update_mappings(Matches, to_db_update)
        session.commit()

    @property 
    def ring(self):
        res = self._from_db(Edges).first()
        if res:
            return res.ring
        return

    @ring.setter
    def ring(self, ring):
        # Setters need a single session and so should not make use of the
        # syntax sugar _from_db
        session = Session()
        res = session.query(Edges).\
               filter(Edges.source == self.source['node_id']).\
               filter(Edges.destination == self.destination['node_id']).first()
        if res:
            res.ring = ring
        else:
            edge = Edges(source=self.source['node_id'],
                         destination=self.destination['node_id'],
                         ring=ring)
            session.add(edge)
            session.commit()
        return        

    @property
    def intersection(self):
        if not hasattr(self, '_intersection'):
            s_fp = self.source.footprint
            d_fp = self.destination.footprint
            self._intersection = s_fp.intersection(d_fp)
        return self._intersection
    
    @property
    def fundamental_matrix(self):
        res = self._from_db(Edges).first()
        if res:
            return res.fundamental
        
    @fundamental_matrix.setter
    def fundamental_matrix(self, v):
        session = Session()
        res = session.query(table_obj).\
               filter(table_obj.source == self.source['node_id']).\
               filter(table_obj.destination == self.destination['node_id']).first()
        if res:
            res.fundamental = v
        else:
            edge = Edges(source=self.source['node_id'],
                         destination=self.destination['node_id'],
                         fundamental = v)
            session.add(edge)
            session.commit()

    def get_overlapping_indices(self, kps):
        ecef = pyproj.Proj(proj='geocent',
			               a=self.parent.config['spatial']['semimajor_rad'],
			               b=self.parent.config['spatial']['semiminor_rad'])
        lla = pyproj.Proj(proj='longlat',
			              a=self.parent.config['spatial']['semiminor_rad'],
			              b=self.parent.config['spatial']['semimajor_rad'])
        lons, lats, alts = pyproj.transform(ecef, lla, kps.xm.values, kps.ym.values, kps.zm.values)
        points = [Point(lons[i], lats[i]) for i in range(len(lons))]
        mask = [i for i in range(len(points)) if self.intersection.contains(points[i])]
        return mask


class NetworkCandidateGraph(network.CandidateGraph):
    node_factory = NetworkNode
    edge_factory = NetworkEdge

    def __init__(self, *args, **kwargs):
        super(NetworkCandidateGraph, self).__init__(*args, **kwargs)
        #self._setup_db_connection()
        self._setup_queues()
        #self._setup_asynchronous_queue_watchers()
        # Job metadata
        self.job_status = defaultdict(dict)

        for i, d in self.nodes(data='data'):
            d.parent = self
        for s, d, e in self.edges(data='data'):
            e.parent = self

        self.processing_queue = config['redis']['processing_queue']

    def _setup_db_connection(self):
        """
        Set up a database connection and session(s)
        """
        try:
            Base.metadata.bind = engine
            Base.metadata.create_all(tables=[Network.__table__, Overlay.__table__,
                                         Edges.__table__, Costs.__table__, Matches.__table__,
                                         Cameras.__table__])
        except ValueError:
            warnings.warn('No SQLAlchemy engine available. Tables not pushed.')

    def _setup_queues(self):
        """
        Setup a 2 queue redis connection for pushing and pulling work/results
        """
        conf = config['redis']
        self.redis_queue = StrictRedis(host=conf['host'],
                                       port=conf['port'],
                                       db=0)

    def _setup_asynchronous_queue_watchers(self, nwatchers=3):
        """
        Setup a sentinel class to watch the results queue
        """
        # Set up the consumers of the 'completed' queue
        for i in range(nwatchers):
            # Set up the sentinel class that watches the registered queues for for messages
            s = AsynchronousQueueWatcher(self, self.redis_queue, config['redis']['completed_queue'])
            s.setDaemon(True)
            s.start()

        # Setup a watcher on the working queue for jobs that fail
        s = AsynchronousFailedWatcher(self, self.redis_queue, config['redis']['working_queue'])
        s.setDaemon(True)
        s.start()

    def empty_queues(self):
        """
        Delete all messages from the redis queue. This a convenience method. 
        The `redis_queue` object is a redis-py StrictRedis object with API
        documented at: https://redis-py.readthedocs.io/en/latest/#redis.StrictRedis
        """
        return self.redis_queue.flushall()

    def apply(self, function, on='edge',out=None, args=(), walltime='01:00:00', **kwargs):
    
        options = {
            'edge' : self.edges,
            'edges' : self.edges,
            'e' : self.edges,
            0 : self.edges,
            'node' : self.nodes,
            'nodes' : self.nodes,
            'n' : self.nodes,
            1 : self.nodes
        }

        # Determine which obj will be called
        onobj = options[on]

        res = []
        key = 1
        if isinstance(on, EdgeView):
            key = 2

        for job_counter, elem in enumerate(onobj.data('data')):
            # Determine if we are working with an edge or a node
            if len(elem) > 2:
                id = (elem[0], elem[1])
                image_path = (elem[2].source['image_path'], 
                              elem[2].destination['image_path'])
            else:
                id = (elem[0])
                image_path = elem[1]['image_path']

            msg = {'id':id,
                    'func':function,
                    'args':args,
                    'kwargs':kwargs,
                    'walltime':walltime,
                    'image_path':image_path,
                    'param_step':1}
                
            self.redis_queue.rpush(self.processing_queue, json.dumps(msg))

        # SLURM is 1 based, while enumerate is 0 based
        job_counter += 1

        # Submit the jobs
        spawn_jobarr('acn_submit', job_counter,
                     mem=config['cluster']['processing_memory'],
                     time=walltime,
                     queue=config['cluster']['queue'],
                     outdir=config['cluster']['cluster_log_dir']+'/slurm-%A_%a.out',
                     env=config['python']['env_name'])
        
        return job_counter
        
    def generic_callback(self, msg):
        id = msg['id']
        if isinstance(id, (int, float, str)):
            # Working with a node
            obj = self.nodes[id]['data']
        else:
            obj = self.edges[id]['data']
            # Working with an edge

        func = msg['func']
        obj.job_status[func]['success'] = msg['success']

        # If the job was successful, no need to resubmit
        if msg['success'] == True:
            return

    def generate_vrts(self, **kwargs):
        for i, n in self.nodes(data='data'):
            n.generate_vrt(**kwargs)

    def compute_overlaps(self):
        query = """
    SELECT ST_AsEWKB(geom) AS geom FROM ST_Dump((
        SELECT ST_Polygonize(the_geom) AS the_geom FROM (
            SELECT ST_Union(the_geom) AS the_geom FROM (
                SELECT ST_ExteriorRing(footprint_latlon) AS the_geom
                FROM images) AS lines
        ) AS noded_lines
    )
)"""
        session = Session()
        oquery = session.query(Overlay)
        iquery = session.query(Images)

        rows = []
        for q in self._engine.execute(query).fetchall():
            overlaps = []
            b = bytes(q['geom']) 
            qgeom = shapely.wkb.loads(b)
            res = iquery.filter(Images.footprint_latlon.ST_Intersects(from_shape(qgeom, srid=949900)))
            for i in res:
                fgeom = to_shape(i.footprint_latlon)
                area = qgeom.intersection(fgeom).area
                if area < 1e-6:
                    continue
                overlaps.append(i.id)
            o = Overlay(geom='SRID=949900;{}'.format(qgeom.wkt), overlaps=overlaps)
            res = oquery.filter(Overlay.overlaps == o.overlaps).first()
            if res is None:
                rows.append(o)
        
        session.bulk_save_objects(rows)
        session.commit()

        res = oquery.filter(sqlalchemy.func.cardinality(Overlay.overlaps) <= 1)
        res.delete(synchronize_session=False)
        session.commit()
        session.close()

    def create_network(self, nodes=[]):
        cmds = 0
        session = Session()
        for res in session.query(Overlay):
            msg = json.dumps({'oid':res.id,'time':time.time()})

            # If nodes are passed, process only those overlaps containing
            # the provided node(s)
            if nodes:
                for r in res.overlaps:
                    if r in nodes:
                        self.redis_queue.rpush(config['redis']['processing_queue'], msg)
                        cmds += 1
                        break
            else:
                self.redis_queue.rpush(config['redis']['processing_queue'], msg)
                cmds += 1
        script = 'acn_create_network'
        spawn_jobarr(script, cmds,
                    mem=config['cluster']['processing_memory'],
                    queue=config['cluster']['queue'],
                    env=config['python']['env_name'])
        session.close()

    @classmethod
    def from_database(cls, query_string='SELECT * FROM public.images'):
        """
        This is a constructor that takes the results from an arbitrary query string,
        uses those as a subquery into a standard polygon overlap query and
        returns a NetworkCandidateGraph object.  By default, an images
        in the Image table will be used in the outer query.

        Parameters
        ----------
        query_string : str
                       A valid SQL select statement that targets the Images table

        Usage
        -----
        Here, we provide usage examples for a few, potentially common use cases.

        ## Spatial Query
        This example selects those images that intersect a given bounding polygon.  The polygon is
        specified as a Well Known Text LINESTRING with the first and last points being the same.
        The query says, select the footprint_latlon (the bounding polygons in the database) that
        intersect the user provided polygon (the LINESTRING) in the given spatial reference system
        (SRID), 949900.

        "SELECT * FROM Images WHERE ST_INTERSECTS(footprint_latlon, ST_Polygon(ST_GeomFromText('LINESTRING(159 10, 159 11, 160 11, 160 10, 159 10)'),949900)) = TRUE"
from_database
        ## Select from a specific orbit
        This example selects those images that are from a particular orbit. In this case,
        the regex string pulls all P##_* orbits and creates a graph from them. This method
        does not guarantee that the graph is fully connected.

        "SELECT * FROM Images WHERE (split_part(path, '/', 6) ~ 'P[0-9]+_.+') = True"

        """
        composite_query = """WITH
	i as ({})
SELECT i1.id as i1_id,i1.path as i1_path, i2.id as i2_id, i2.path as i2_path
FROM
	i as i1, i as i2
WHERE ST_INTERSECTS(i1.footprint_latlon, i2.footprint_latlon) = TRUE
AND i1.id < i2.id""".format(query_string)
        session = Session()
        res = session.execute(composite_query)

        adjacency = defaultdict(list)
        adjacency_lookup = {}
        for r in res:
            sid, spath, did, dpath = r

            adjacency_lookup[spath] = sid
            adjacency_lookup[dpath] = did
            if spath != dpath:
                adjacency[spath].append(dpath)
        session.close()
        # Add nodes that do not overlap any images
        obj = cls.from_adjacency(adjacency, node_id_map=adjacency_lookup, config=config)
    
        return obj

    @classmethod
    def from_filelist(cls, filelist, basepath=None):
        """
        This methods instantiates a network candidate graph by first parsing
        the filelist and adding those images to the database. This method then
        dispatches to the from_database cls method to create the network
        candidate graph object.
        """
        if isinstance(filelist, str):
            filelist = io_utils.file_to_list(filelist)
        
        if basepath:
            filelist = [(f, os.path.join(basepath, f)) for f in filelist]
        else:
            filelist = [(os.path.basename(f), f) for f in filelist]
        
        parent = Parent(config)
        # Get each of the images added to the DB (duplicates, by PATH, are omitted)
        for f in filelist:
            n = NetworkNode(image_name=f[0], image_path=f[1], parent=parent)
        pathlist = [f[1] for f in filelist]

        qs = 'SELECT * FROM public.Images WHERE public.Images.path IN ({})'.format(','.join("'{0}'".format(p) for p in pathlist))
        return NetworkCandidateGraph.from_database(query_string=qs)

class AsynchronousQueueWatcher(threading.Thread):

    def __init__(self, parent, queue, name):
        """len(k)
        Parameters
        ----------
        parent : obj
                 The parent object with callback funcs that this class
                 dispathces to when a queue has messages.

        queues : dict
                 with key as the queue name and value as the callback function
                 name

        """
        super(AsynchronousQueueWatcher, self).__init__()
        self.queue = queue
        self.parent = parent
        self.name = name

    def run(self):

        while True:
            msg = self.queue.lpop(self.name)  # or blpop?
            if msg:
                callback_func = self.parent.generic_callback(json.loads(msg))

class AsynchronousFailedWatcher(threading.Thread):

    def __init__(self, parent, queue, name):
        """len(k)
        Parameters
        ----------
        parent : obj
                 The parent object with callback funcs that this class
                 dispathces to when a queue has messages.

        queues : dict
                 with key as the queue name and value as the callback function
                 name

        """
        super(AsynchronousFailedWatcher, self).__init__()
        self.queue = queue
        self.parent = parent
        self.name = name

    def run(self):
        #try:
        while True:
            msgs = self.queue.lrange(self.name, 0, -1)
            to_pop_and_resubmit = []
            t = time.time()
            for msg in msgs:
                msg = json.loads(msg)
                if  t > msg['max_time'] + 30:  # 10 is the approx buffer that slurm offers
                    to_pop_and_resubmit.append(msg)

            # Remove the message from the work queue is it is expired.
            for msg in to_pop_and_resubmit:
                callback_func = getattr(self.parent, 'generic_callback')
                self.queue.lrem(self.name,0, json.dumps(msg))
                self.parent.generic_callback(msg)

class SubCandidateGraph(nx.graphviews.SubGraph, NetworkCandidateGraph):
    def __init__(self, *args, **kwargs):
        super(SubCandidateGraph, self).__init__(*args, **kwargs)
        #self._setup_db_connection()
        self._setup_queues()
        self.processing_queue = config['redis']['processing_queue']

nx.graphviews.SubGraph = SubCandidateGraph
