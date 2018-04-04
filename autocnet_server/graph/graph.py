from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
import os
import pickle
import time
import threading

from sqlalchemy.orm import aliased, create_session, scoped_session, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine
from geoalchemy2.shape import to_shape
from geoalchemy2.elements import WKTElement
from shapely.geometry import Point

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
    sys.path.insert(0, acp)

from autocnet.graph import node, network, edge
from autocnet.utils import utils
from autocnet.io import keypoints as io_keypoints
from autocnet.transformation.fundamental_matrix import compute_reprojection_error

from autocnet_server.camera.csm_camera import create_camera, ecef_to_latlon
from autocnet_server.camera import generate_vrt
from autocnet_server.cluster.slurm import spawn, spawn_jobarr
from autocnet_server.db.model import Images, Keypoints, Matches, Cameras, Network, Base, Overlay, Edges
from autocnet_server.db.connection import db_connect

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
        outpath = self.parent.config['directories']['vrt_dir']
        generate_vrt.warped_vrt(self.camera, self.geodata.raster_size,
                                self.geodata.file_name, outpath=outpath)
    #def _clean(self):
    #    pass

class NetworkEdge(edge.Edge):
    def __init__(self, *args, **kwargs):
        super(NetworkEdge, self).__init__(*args, **kwargs)
        self.job_status = defaultdict(dict)
        self.default_status = {'submission': None, 'count':0, 'success':False}

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

    def ring_match(self, overlap=True, target_points=25, tolerance=0.01):
        # Create the message that parameterizes the job
        msg = {'sidx':self.source['node_id'],
               'didx':self.destination['node_id'],
               'time':time.time(),
               'target_points':target_points,
               'tolerance':tolerance}

        # Update the job status
        if 'ring_match' not in self.job_status.keys():
            self.job_status['ring_match'] = {}
            self.job_status['ring_match']['submission'] = msg
            self.job_status['ring_match']['count'] = 0
            self.job_status['ring_match']['success'] = False
        # Put the message onto the redis queue for processing
        msg = json.dumps(msg)
        self.parent.redis_queue.rpush(config['redis']['processing_queue'], msg)

        return True


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
        self._connection = self._engine.connect()
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
            s = AsynchronousQueueWatcher(self, self.redis_queue)
            s.setDaemon(True)
            s.start()

    def generate_vrts(self, **kwargs):
        for i, n in self.nodes(data='data'):
            n.generate_vrt(**kwargs)

    def ring_match(self, overlap=True, edges=[], time='01:00:00'):
        cmds = []
        if edges:
            for e in edges:
                cmds.append(self.edges[e]['data'].ring_match(overlap=overlap))
        else:
            for s, d, e in self.edges(data='data'):
                cmds.append(e.ring_match(overlap=overlap))
        # TODO: This should not be hard coded, but pulled from the install location
        script = '/home/jlaura/autocnet_server/bin/ring_match.py'
        spawn_jobarr(config['python']['pybin'], script,
                     len(cmds), mem=config['cluster']['processing_memory'],
                     time=time)

        return True

    def ring_matcher_callback(self, msg):
        source = msg['source']
        destination = msg['destin']

        # Pull the correct edge and dispatch to generate the match objs
        e = self.edges[source, destination]['data']
        if msg['success']:
            e.job_status['ring_match']['success'] = True
        else:
            if e.job_status['ring_match']['count'] <= config['cluster']['maxfailures']:
                e.job_status['ring_match']['count'] += 1
                target_points = msg['target_points']
                tolerance = msg['tolerance']
                target_points -= 5
                tolerance += 0.01
                if target_points < 15:
                    break
                # Respawn the job using the normal slurm method
                e.ring_match(target_points=target_points, tolerance=tolerance)
                # This hard coded command is poor form in that this is making the ring_matcher
                # now parameterize differently, what we want to do here is think about adding
                # some devision making to the system - if matching well fails, reduce either
                # the threshold tolerance or the number of required 'good' points
                cmd = '{} /home/jlaura/autocnet_server/bin/ring_match.py'.format(config['python']['pybin'])
                spawn(cmd, mem=config['cluster']['processing_memory'])

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
        script = '/home/jlaura/autocnet_server/bin/create_network.py'
        spawn_jobarr(config['python']['pybin'], script, cmds, mem=config['cluster']['processing_memory'])

    @classmethod
    def from_database(cls, query_string='SELECT * FROM Images'):
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
    	db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(config['database']['database_username'],
                                                      config['database']['database_password'],
            	                                      config['database']['database_host'],
                                                      config['database']['pgbouncer_port'],
                                                      config['database']['database_name'])        
        composite_query = """WITH 
	i as ({})
SELECT i1.id as i1_id,i1.path as i1_path, i2.id as i2_id, i2.path as i2_path
FROM
	i as i1, i as i2
WHERE ST_INTERSECTS(i1.footprint_latlon, i2.footprint_latlon) = TRUE
AND i1.id < i2.id""".format(query_string)
        engine = create_engine(db_uri)
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

    def __init__(self, parent, queue):
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

    def run(self):
        #try:
        while True:
            msg = self.queue.lpop(config['redis']['completed_queue'])  # or blpop?
            if msg:
                msg = json.loads(msg)
                callback_func = getattr(self.parent, msg['callback'])
                callback_func(msg)
            else:
                time.sleep(5)  # Does this need to sleep?
        #except:
        #    print('err: ', msg)
