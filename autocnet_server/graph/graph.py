from collections import defaultdict
from itertools import combinations
import json
import os
import pickle
import time
import threading

from concurrent.futures import ThreadPoolExecutor

from autocnet.graph import node, network, edge
from autocnet.utils import utils
from autocnet.io import keypoints as io_keypoints
from autocnet.transformation.fundamental_matrix import compute_reprojection_error

from autocnet_server.camera.csm_camera import create_camera, ecef_to_latlon
from autocnet_server.camera import generate_vrt
from autocnet_server.cluster.slurm import spawn, spawn_jobarr
from autocnet_server.db.model import Images, Keypoints, Matches, Cameras, Network, Base, Overlay
from autocnet_server.db.connection import db_connect
from autocnet_server.config import AutoCNet_Config

from sqlalchemy.orm import aliased, create_session, scoped_session, sessionmaker
from sqlalchemy import create_engine
from geoalchemy2.shape import to_shape
from geoalchemy2.elements import WKTElement
from shapely.geometry import Point

import numpy as np
import pandas as pd
import pyproj

import hotqueue as hq

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
        outpath = self.parent.config.vrt_dir
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
        ecef = pyproj.Proj(proj='geocent', a=self.parent.config.semimajor_rad, b=self.parent.config.semiminor_rad)
        lla = pyproj.Proj(proj='longlat', a=self.parent.config.semiminor_rad, b=self.parent.config.semimajor_rad)
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

    def ring_match(self, overlap=True):
        # Create the message that parameterizes the job
        spath = self.source.keypoint_file
        dpath = self.destination.keypoint_file

        msg = {'sidx':self.source['node_id'], 'didx':self.destination['node_id'],
                'spath':spath, 'dpath':dpath}

        # Update the job status
        if 'ring_match' not in self.job_status.keys():
            self.job_status['ring_match'] = {}
            self.job_status['ring_match']['submission'] = msg
            self.job_status['ring_match']['count'] = 0
            self.job_status['ring_match']['success'] = False
        # Put the message onto the redis queue for processing
        self.parent.processing_queue.put(msg)

        return True

    def add_matches(self, d):
        self['ring'] = d['ring']
        source_idx = d['sidx']
        destin_idx = d['didx']
        s = self.source
        d = self.destination
        matches = []
        skps = s.get_keypoints(index=source_idx)
        dkps = d.get_keypoints(index=destin_idx)
        #skps and dkps will come out sorted by the indices

        for i, j in zip(source_idx, destin_idx):
            sidx = int(i)
            didx = int(j)
            sx = float(skps.loc[i][0])
            sy = float(skps.loc[i][1])
            dx = float(dkps.loc[j][0])
            dy = float(dkps.loc[j][1])
            # Use the source camera to project the sx, sy to ground
            gnd = s.camera.imageToGround(sy, sx, 0)
            lon, lat, alt = ecef_to_latlon(gnd)
            # TODO: Hard coded srid needs to be set at the project level
            geom = 'SRID=949900;POINTZ({} {} {})'.format(lon, lat, alt)
            m = Matches(source=s['node_id'], source_idx=sidx,
                        destination=d['node_id'], destination_idx=didx,
                        lat=float(lat), lon=float(lon), geom=geom,
                        source_x=sx, source_y=sy,
                        destination_x=dx, destination_y=dy)
            matches.append(m)
        return matches

class NetworkCandidateGraph(network.CandidateGraph):
    node_factory = NetworkNode
    edge_factory = NetworkEdge

    def __init__(self, *args, config=AutoCNet_Config, **kwargs):
        super(NetworkCandidateGraph, self).__init__(*args, **kwargs)
        self.config = config()
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
        db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(self.config.database_username,
                                                      self.config.database_password,
                                                      self.config.database_host,
                                                      self.config.database_port,
                                                      self.config.database_name)
        self._engine = create_engine(db_uri)
        self._connection = self._engine.connect()
        self.session = scoped_session(sessionmaker(bind=self._engine))

        # TODO: This adds the tables to the db if they do not exist already
        Base.metadata.bind = self._engine
        Base.metadata.create_all(tables=[Network.__table__, Overlay.__table__])

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
        self.processing_queue = hq.HotQueue("processor",
                                            serializer=json,
                                            host="smalls",
                                            port=8000,
                                            db=0)
        self.completed_queue = hq.HotQueue("completed",
                                           serializer=json,
                                           host="smalls",
                                           port=8000,
                                           db=0)

    def _setup_asynchronous_queue_watchers(self, nwatchers=3):
        """
        Setup a sentinel class to watch the results queue
        """
        for i in range(nwatchers):
            # Set up the sentinel class that watches the registered queues for for messages
            s = AsynchronousQueueWatcher(self, self.completed_queue)
            s.setDaemon(True)
            s.start()

    def generate_vrts(self, **kwargs):
        for i, n in self.nodes(data='data'):
            n.generate_vrt(**kwargs)

    def ring_match(self, overlap=True, edges=[]):
        cmds = []
        if edges:
            for e in edges:
                cmds.append(self.edges[e]['data'].ring_match(overlap=overlap))
        else:
            for s, d, e in self.edges(data='data'):
                cmds.append(e.ring_match(overlap=overlap))
        # TODO: This should not be hard coded, but pulled from the install location
        script = '/home/jlaura/autocnet_server/bin/ring_match.py'
        spawn_jobarr(self.config.pybin, script,
                     len(cmds), mem=self.config.processing_memory)

        return True

    def ring_matcher_callback(self, msg):
        source = msg['source']
        destination = msg['destin']

        # Pull the correct edge and dispatch to generate the match objs
        e = self.edges[source, destination]['data']
        if msg['success']:
            matches = e.add_matches(msg)

            # Bulk insert the matches into the db
            self.session.bulk_save_objects(matches)
            self.session.commit()
            e.job_status['ring_match']['success'] = True
        else:
            if 'ring_match' not in e.job_status.keys():
                cmd = '{} /home/jlaura/autocnet_server/bin/ring_match.py -p 20'.format(self.config.pybin)
                spawn(cmd, mem=self.config.processing_memory)

            elif e.job_status['ring_match']['count'] <= self.config.maxfailures:
                e.job_status['ring_match']['count'] += 1
                # Respawn the job using the normal slurm method
                e.ring_match()
                # This hard coded command is poor form in that this is making the ring_matcher
                # now parameterize differently, what we want to do here is think about adding
                # some devision making to the system - if matching well fails, reduce either
                # the threshold tolerance or the number of required 'good' points
                cmd = '{} /home/jlaura/autocnet_server/bin/ring_match.py -p 20'.format(self.config.pybin)
                spawn(cmd, mem=self.config.processing_memory)

    def create_network(self):
        oquery = self.session.query(Overlay)
        mquery = self.session.query(Matches)
        kquery = self.session.query(Keypoints)

        def check_in(r, poly):
            p = to_shape(r.geom)
            return p.within(poly)

        cmds = 0
        for res in oquery:
            msg = {}

            poly = to_shape(res.geom)
            overlaps = res.overlaps

            msg['oid'] = res.id
            msg['poly'] = poly.wkt
            msg['overlaps'] = res.overlaps
            files = kquery.filter(Keypoints.image_id.in_(res.overlaps)).all()
            files = {i.image_id:i.path for i in files}

            msg['files'] = files
            msg['matches'] = {}

            # Pulling all these matches is a serial bottleneck...
            for e in combinations(res.overlaps, 2):
                edge = self.edges[e]['data']
                m = edge.matches
                if len(m) > 0:
                    msg['matches'][str(e)] = m.to_json(double_precision=15)
            self.processing_queue.put(msg)
            cmds += 1
            if cmds % 100 == 0:
                script = '/home/jlaura/autocnet_server/bin/create_network.py'
                spawn_jobarr(self.config.pybin, script, cmds, mem=self.config.processing_memory)
                cmds = 0
        script = '/home/jlaura/autocnet_server/bin/create_network.py'
        spawn_jobarr(self.config.pybin, script, cmds, mem=self.config.processing_memory)
        return

    def create_network_callback(self, msg):
        if msg['success']:
            pts = msg['points']
            to_add = []
            for p in msg['points']:
                n = Network(image_id=p['image_id'],
                            keypoint_id=p.get('keypoint_id', None),
                            x = p['x'], y = p['y'],
                            match_id = p.get('match_id', None),
                            point_id = p.get('point_id', None),
                            geom = p['geom'])
                to_add.append(n)
            self.session.bulk_save_objects(to_add)
            self.session.commit()




    @classmethod
    def from_database(cls, config=AutoCNet_Config):
        sc = config()
        db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(sc.database_username,
                                                      sc.database_password,
                                                      sc.database_host,
                                                      sc.database_port,
                                                      sc.database_name)
        engine = create_engine(db_uri)
        connection = engine.connect()
        session = sessionmaker(bind=engine)()

        # Add images that overlap
        image_alias = aliased(Images)
        query = session.query(Images, image_alias)
        res = query.filter(Images.id <= image_alias.id).filter(Images.footprint_latlon.ST_Intersects(image_alias.footprint_latlon))
        adjacency = defaultdict(list)
        adjacency_lookup = {}
        for r in res:
            s = r[0]
            d = r[1]
            adjacency_lookup[s.path] = s.id
            adjacency_lookup[d.path] = d.id
            if s.path != d.path:
                adjacency[s.path].append(d.path)

        # Add nodes that do not overlap any images
        obj = cls.from_adjacency(adjacency, node_id_map=adjacency_lookup, config=config)
        session.close()
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
        for msg in self.queue.consume():
            callback_func = getattr(self.parent, msg['callback'])
            callback_func(msg)
        #except:
        #    print('err: ', msg)

class NetworkControlNetwork():
    def __init__(self, config=AutoCNet_Config):
        self.config = config()
        #TODO: Just use a mixin here since copied from NCG.
        self._setup_db_connection()
        self._setup_queues()
        self._setup_asynchronous_queue_watchers()

    def _from_db(self, table_obj, key='image_id', value=None):
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
        return self.session.query(table_obj).filter(getattr(table_obj,key) == value).first()

    def _setup_db_connection(self):
        """
        Set up a database connection and session(s)
        """
        db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(self.config.database_username,
                                                      self.config.database_password,
                                                      self.config.database_host,
                                                      self.config.database_port,
                                                      self.config.database_name)
        self._engine = create_engine(db_uri)
        self._connection = self._engine.connect()
        self.session = scoped_session(sessionmaker(bind=self._engine))

        # TODO: This adds the tables to the db if they do not exist already
        Base.metadata.bind = self._engine
        Base.metadata.create_all(tables=[Network.__table__, Overlay.__table__])

    def _setup_queues(self):
        """
        Setup a 2 queue redis connection for pushing and pulling work/results
        """
        # TODO: Remove hard coding and graph from config
        self.processing_queue = hq.HotQueue("processor",
                                            serializer=json,
                                            host="smalls",
                                            port=8000,
                                            db=0)
        self.completed_queue = hq.HotQueue("completed",
                                           serializer=json,
                                           host="smalls",
                                           port=8000,
                                           db=0)

    def _setup_asynchronous_queue_watchers(self, nwatchers=3):
        """
        Setup a sentinel class to watch the results queue
        """
        for i in range(nwatchers):
            # Set up the sentinel class that watches the registered queues for for messages
            s = AsynchronousQueueWatcher(self, self.completed_queue)
            s.setDaemon(True)
            s.start()

    def generate_overlays(self):
        # TODO: Method to generate overlays
        pass

    def create_network(self):
        oquery = self.session.query(Overlay)
        mquery = self.session.query(Matches)

        def check_in(r, poly):
            p = to_shape(r.geom)
            return p.within(poly)

        fundamentals = {}
        for res in oquery:
            if len(res.overlaps) == 2:
                # Run the suppression algorithm and write to the network obj. (or REDIS)
                pass
            else:
                poly = to_shape(res.geom)
                overlaps = res.overlaps
                # Case n > 2 images
                matches = []
                # Merge together all of the points from all of the images and grab the fundamental matrices
                for e in combinations(res.overlaps, 2):
                    edge = ncg.edges[e]['data']
                    edge.compute_fundamental_matrix(method='ransac', reproj_threshold=20)
                    fundamentals[e] = edge['fundamental_matrix']
                    m = edge.matches

                    err = compute_reprojection_error(edge['fundamental_matrix'],
                                                    make_homogeneous(m[['source_x', 'source_y']].values),
                                                    make_homogeneous(m[['destination_x', 'destination_y']].values))

                    m['strength'] = err
                    matches.append(m)


                matches = pd.concat(matches)

                # Of the concatenated matches only a subset intersect the geometry for this overlap, pull these
                intersects = matches.apply(check_in, args=(poly,), axis=1)
                matches = matches[intersects]
                break
