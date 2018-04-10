from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
import os
import pickle
import sys
import time
import threading

from sqlalchemy.orm import aliased, create_session, scoped_session, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine, func

from geoalchemy2.shape import to_shape, from_shape
from geoalchemy2.elements import WKTElement, WKBElement
from shapely.geometry import Point
import shapely

import numpy as np
import pandas as pd
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
from autocnet.utils import utils
from autocnet.io import keypoints as io_keypoints
from autocnet.transformation.fundamental_matrix import compute_reprojection_error

from autocnet_server.camera.csm_camera import create_camera, ecef_to_latlon
from autocnet_server.camera import generate_vrt
from autocnet_server.cluster.slurm import spawn, spawn_jobarr
from autocnet_server.db.model import Images, Keypoints, Matches, Cameras, Network, Base, Overlay, Edges
from autocnet_server.db.connection import new_connection
from autocnet_server.utils.utils import slurm_walltime_to_seconds

class NetworkNode(node.Node):
    def __init__(self, *args, **kwargs):
        super(NetworkNode, self).__init__(*args, **kwargs)

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
        return self.parent.session.query(table_obj).filter(getattr(table_obj,key) == self['node_id']).first()

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
        if not hasattr(self, '_camera'):
            res = self._from_db(Cameras)
            self._camera = pickle.loads(res.camera)
        return self._camera

    def get_keypoints(self, index=None, format='hdf', overlap=False, homogeneous=False, **kwargs):
        """
        Return the keypoints for the node.  If index is passed, return
        the appropriate subset.
        Parameters
        ----------
        index : iterable
                indices for of the keypoints to return
        Returns
        -------
         : dataframe
           A pandas dataframe of keypoints
        """
        path = self.keypoint_file

        if format == 'npy':
            kps = io_keypoints.from_npy(path)
        elif format == 'hdf':
            kps = io_keypoints.from_hdf(path, index=index, descriptors=False, **kwargs)

        kps = kps[['x', 'y']]  # Added for fundamental

        if homogeneous:
            # TODO: Make the kps homogeneous
            pass

        return kps

    @property
    def keypoint_file(self):
        res = self._from_db(Keypoints)
        return res.path

    @property
    def footprint(self):
        res = self.parent.session.query(Images).filter(Images.id == self['node_id']).first()
        return to_shape(res.footprint_latlon)

    def get_descriptors(self, index=None):
        path = self.keypoint_file
        if self.descriptors is None:
            self.load_features(path, format='hdf')
        if index is not None:
            desc = self.descriptors[index]
        else:
            desc = self.descriptors
        self.decriptors = None
        return desc

    def generate_vrt(self, **kwargs):
        """
        Using the image footprint, generate a VRT to that is usable inside
        of a GIS (QGIS or ArcGIS) to visualize a warped version of this image.
        """
        outpath = config['directories']['vrt_dir']
        generate_vrt.warped_vrt(self.camera, self.geodata.raster_size,
                                self.geodata.file_name, outpath=outpath)
    #def _clean(self):
    #    pass

class NetworkEdge(edge.Edge):
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
        return self.parent.session.query(table_obj).\
               filter(table_obj.source == self.source['node_id'],
               filter(table_obj.destination == self.destination['node_id']))

    @property
    def matches(self):
        q = self.parent.session.query(Matches)
        qf = q.filter(Matches.source == self.source['node_id'],
                      Matches.destination == self.destination['node_id'])
        return pd.read_sql(qf.statement, q.session.bind)

    @property
    def intersection(self):
        if not hasattr(self, '_intersection'):
            s_fp = self.source.footprint
            d_fp = self.destination.footprint
            self._intersection = s_fp.intersection(d_fp)
        return self._intersection

    #def clean(self):
    #    pass

    def get_overlapping_indices(self, kps):
        ecef = pyproj.Proj(proj='geocent',
			   a=self.parent.config['spatial']['semimajor_rad'],
			   b=self.parent.config['spatial']['semiminor_rad'])
        lla = pyproj.Proj(proj='longlat',
			  a=self.parent.config['spatial']['semiminor_rad'],
			  b=self.parent.config['spatial']['.semimajor_rad'])
        lons, lats, alts = pyproj.transform(ecef, lla, kps.xm.values, kps.ym.values, kps.zm.values)
        points = [Point(lons[i], lats[i]) for i in range(len(lons))]
        mask = [i for i in range(len(points)) if self.intersection.contains(points[i])]
        return mask

    @utils.methodispatch
    def get_keypoints(self, node, index=None, homogeneous=False, overlap=False):
        if not hasattr(index, '__iter__') and index is not None:
            raise TypeError
        kps = node.get_keypoints(index=index, overlap=overlap)
        if overlap:
            indices = self.get_overlapping_indices(kps)
            kps = kps.iloc[indices]
        return kps

    @get_keypoints.register(str)
    def _(self, node, index=None, homogeneous=False, overlap=False):
        if not hasattr(index, '__iter__') and index is not None:
            raise TypeError
        node = node.lower()
        node = getattr(self, node)
        return self.get_keypoints(node, index=index, homogeneous=homogeneous, overlap=overlap)

    def ring_match(self):
        if len(self.job_status['ring_match']) == 0:
            parameters = config['algorithms']['ring_match'][0]
            self.job_status['ring_match'] = {'sidx':self.source['node_id'],
                                             'didx':self.destination['node_id'],
                                             'time':'',
                                             'task':'ring_match',
                                             'success':False,
                                             'param_step':0,
                                             'count':0,
                                             'target_points':parameters['target_points'],
                                             'tolerance':parameters['tolerance']}

        return self.job_status['ring_match']


class NetworkCandidateGraph(network.CandidateGraph):
    node_factory = NetworkNode
    edge_factory = NetworkEdge

    def __init__(self, *args, **kwargs):
        super(NetworkCandidateGraph, self).__init__(*args, **kwargs)
        self._setup_db_connection()
        self._setup_queues()
        self._setup_asynchronous_queue_watchers()

        # Job metadata
        self.job_status = defaultdict(dict)

        for i, d in self.nodes(data='data'):
            d.parent = self
        for s, d, e in self.edges(data='data'):
            e.parent = self

    def __key(self):
        # TODO: This needs to be a real self identifying key
        return 'abcde'

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return type(self) == type(other) and self.__key() == other.__key()

    def _setup_db_connection(self):
        """
        Set up a database connection and session(s)
        """

        db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(config['database']['database_username'],
                                                      config['database']['database_password'],
                                                      config['database']['database_host'],
                                                      config['database']['database_port'],
                                                      config['database']['database_name'])
        self._engine = create_engine(db_uri, pool_size=2)
        #self._connection = self._engine.connect()
        self.session = scoped_session(sessionmaker(bind=self._engine))

        # TODO: This adds the tables to the db if they do not exist already
        Base.metadata.bind = self._engine
        Base.metadata.create_all(tables=[Network.__table__, Overlay.__table__, Edges.__table__])

    @property
    def unmatched_edges(self):
        """
        Returns a list of edges (source, destination) that do not have
        entries in the matches dataframe.
        """
        #TODO: This is slow as it hits the database once for each set, optimization needed.
        unmatched = []
        for s, d, e in self.edges(data='data'):
            if len(e.matches) == 0:
                unmatched.append((s,d))

        return unmatched

    def _setup_queues(self):
        """
        Setup a 2 queue redis connection for pushing and pulling work/results
        """
        # TODO: Remove hard coding and graph from config
        self.redis_queue = StrictRedis(host='smalls',
                                       port=8000,
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
        oquery = self.session.query(Overlay)
        iquery = self.session.query(Images)

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
        
            rows.append(o)
        self.session.bulk_save_objects(rows)
        self.session.commit()

        res = oquery.filter(func.cardinality(Overlay.overlaps) <= 1)
        res.delete(synchronize_session=False)
        self.session.commit()

    def ring_match(self, edges=[], walltime='01:00:00'):
        t = time.time()
        if edges:
            for job_counter, e in enumerate(edges):
                msg = self.edges[e]['data'].ring_match()
                msg['walltime'] = walltime
                msg['callback'] = 'ring_matcher_callback'
                # Put the message onto the redis queue for processing
                self.redis_queue.rpush(config['redis']['processing_queue'], json.dumps(msg))
        else:
            for job_counter, (s, d, e) in enumerate(self.edges(data='data')):
                msg = e.ring_match()
                msg['walltime'] = walltime
                msg['callback'] = 'ring_matcher_callback'
                # Put the message onto the redis queue for processing
                self.redis_queue.rpush(config['redis']['processing_queue'], json.dumps(msg))
        job_counter += 1 # Slurm counter is 1 based.
        # TODO: This should not be hard coded, but pulled from the install location
        script = '/home/acpaquette/repos/autocnet_server/bin/ring_match.py'
        spawn_jobarr(config['python']['pybin'], script,job_counter,
                     mem=config['cluster']['processing_memory'],
                     time=walltime, queue=config['cluster']['queue'])

        return job_counter


    def ring_matcher_callback(self, msg):
        source = msg['sidx']
        destination = msg['didx']

        # Pull the correct edge and dispatch to generate the match objs
        e = self.edges[source, destination]['data']
        rm = e.job_status['ring_match']
        if msg['success']:
            rm['success'] = True
        else:
            if rm['count'] <= config['cluster']['maxfailures']:
                rm['count'] += 1
                rm['param_step'] += 1
                if rm['param_step'] >= len(config['algorithms']['ring_match']):
                    # All parameter combinations are exhausted
                    return
                # Increment the parameter space defined in the config
                current_param_step = rm['param_step']
                parameters = config['algorithms']['ring_match'][current_param_step]
                rm['target_points'] = parameters['target_points']
                rm['tolerance'] = parameters['tolerance']
                rm['time'] = time.time()
                # Respawn the job using the normal slurm method
                self.redis_queue.rpush(config['redis']['processing_queue'], json.dumps(rm))
                # This hard coded command is poor form in that this is making the ring_matcher
                # now parameterize differently, what we want to do here is think about adding
                # some devision making to the system - if matching well fails, reduce either
                # the threshold tolerance or the number of required 'good' points
                cmd = '{} /home/acpaquette/repos/autocnet_server/bin/ring_match.py'.format(config['python']['pybin'])
                spawn(cmd, mem=config['cluster']['processing_memory'],
                      time=msg['walltime'], queue=config['cluster']['queue'])

    def create_network(self, nodes=[]):
        cmds = 0
        for res in self.session.query(Overlay):
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
        script = '/home/acpaquette/repos/autocnet_server/bin/create_network.py'
        spawn_jobarr(config['python']['pybin'], script, cmds,
                    mem=config['cluster']['processing_memory'],
                    queue=config['cluster']['queue'])

    @classmethod
    def from_database(cls, query_string='SELECT * FROM public.Images'):
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
        _, engine = new_connection()
        res = engine.execute(composite_query)

        adjacency = defaultdict(list)
        adjacency_lookup = {}
        for r in res:
            sid, spath, did, dpath = r

            adjacency_lookup[spath] = sid
            adjacency_lookup[dpath] = did
            if spath != dpath:
                adjacency[spath].append(dpath)

        # Add nodes that do not overlap any images
        obj = cls.from_adjacency(adjacency, node_id_map=adjacency_lookup, config=config)
        return obj

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
                msg = json.loads(msg)
                callback_func = getattr(self.parent, msg['callback'])
                callback_func(msg)

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
                callback_func = getattr(self.parent, msg['callback'])
                self.queue.lrem(self.name,0, json.dumps(msg))
                callback_func(msg)
