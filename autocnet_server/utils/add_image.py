from collections import defaultdict
import glob
import json
import os
import socket

from autocnet_server.cluster.slurm import spawn
from autocnet_server.db.connection import db_connect
from autocnet_server.db.model import Images, Keypoints, Matches, Cameras
from autocnet_server.config import AutoCNet_Config
from geoalchemy2.elements import WKTElement

from sqlalchemy import create_engine
from sqlalchemy.orm import create_session, scoped_session, sessionmaker

from shapely.geometry import shape

import Pyro4
from threading import Thread

class ImageAdder():
    def __init__(self, config=AutoCNet_Config):
        self.config = config()
        db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(self.config.database_username,
                                                      self.config.database_password,
                                                      self.config.database_host,
                                                      self.config.database_port,
                                                      self.config.database_name)
        self._engine = create_engine(db_uri)
        self._connection = self._engine.connect()
        self._session = sessionmaker(bind=self._engine)()
        self._daemon = Pyro4.Daemon()
        self._thread = Thread(target=self._daemon.serveSimple,
                              args=({self : self.config.image_adder_uri},),
                              kwargs={'ns':False,
                                      'port':self.config.image_adder_port,
                                      'host':self.config.image_adder_host},
                              daemon=True)
        self._thread.start()
        self.job_status = defaultdict(dict)

    def extract(self, path, force=False, **kwargs):
        """

        """
        if not force:
            res = self._session.query(Images).filter(Images.path == path).first()
            if res:
                return 'Image already processed'

        hostname = socket.gethostname()
        callback_uri = 'PYRO:{}@{}:{}'.format(self.config.image_adder_uri,
                                               hostname,
                                               self.config.image_adder_port)

        command = '{} /home/jlaura/autocnet_server/bin/extract_features.py {} {}'
        command = command.format(self.config.pybin, path, callback_uri)
        if self.config.cluster_log_dir is not None:
            log_out = self.config.cluster_log_dir + '/%j.log'
        else:
            out = '%j.log'

        # Spawn the job and update the submission tracker
        res = spawn(command, out=log_out, mem=self.config.extractor_memory)
        self.job_status[path]['submission'] = res
        self.job_status[path]['count'] = 0

    def add_images(self, path, extension, force=False, append=False, **kwargs):
        """
        Add a directory of images to the processing queue to have features extracted.
        This method extracts all imags in the path with a given extension.  If force
        is set to True, images that already exist in the database will have features
        re-extracted.  If append is also True, existing features will be preserved
        and newly extracted features appended to the existing set.  The kwargs are
        passed to cluster.slurm.spawn.

        Parameters
        ----------
        path : string
               The directory containing images to be extracted

        extension : string
                    The file extension to search for in the directory, e.g., '.cub'

        force : boolean
                If True re-extract keypoints for images already in the database
                Default: False

        append : boolean
                 If True, append newly extracted keypoints to the existing set. If
                 False (default) replace the existing keypoints with a new set
                 of keypoints.
        """
        #TODO: Add support for append.
        search_dir = os.path.join(path, '*.{}'.format(extension))
        files = glob.glob(search_dir)
        results = []
        for f in files:
            res = self.add_image(f, force=force)
            results.append(res)

    def add_image(self, path, force=False):
        return self.extract(path, force=force)

    @property
    def failed_jobs(self):
        return [k for k, v in self.job_status.items() if v != True]

    def rerun_failed_jobs(self):
        files = self.failed_jobs
        self.add_images(files, force=True)

    @Pyro4.expose
    def add_image_callback(self, d):
        if d['success']:
            self.job_status[d['path']]['success'] = True
        else:
            self.job_status[d['path']]['count'] += 1
            # Job failed, try again, up to 3 times
            if self.job_status[d['path']]['count'] <= self.config.maxfailures:
                self.extract(d['path'], force=True)
