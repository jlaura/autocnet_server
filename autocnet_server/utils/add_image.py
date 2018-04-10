from collections import defaultdict
import glob
import json
import os
import socket
import yaml

from autocnet_server.cluster.slurm import spawn
from autocnet_server.db.model import Images, Keypoints, Matches, Cameras
<<<<<<< HEAD
from autocnet_server.db.connection import new_connection
from autocnet_server import config
=======
>>>>>>> dd25bc9ce3d488b2e515e75b35c2a8b2e8f58714
from geoalchemy2.elements import WKTElement

from sqlalchemy import create_engine
from sqlalchemy.orm import create_session, scoped_session, sessionmaker

from shapely.geometry import shape

import Pyro4
from threading import Thread

with open(os.environ['autocnet_config'], 'r') as f:
    config = yaml.load(f)

class ImageAdder():
    def __init__(self):
<<<<<<< HEAD
        db_uri, self._engine = new_connection()
=======

        db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(config['database']['database_username'],
                                                                                   config['database']['database_password'],
                                                                                   config['database']['database_host'],
                                                                                   config['database']['database_port'],
                                                                                   config['database']['database_name'])
        self._engine = create_engine(db_uri)
>>>>>>> dd25bc9ce3d488b2e515e75b35c2a8b2e8f58714
        self._connection = self._engine.connect()
        self._session = sessionmaker(bind=self._engine)()
        self._daemon = Pyro4.Daemon()
        self._thread = Thread(target=self._daemon.serveSimple,
                              args=({self : config['pyro']['image_adder_uri']},),
                              kwargs={'ns':False,
                                      'port':config['pyro']['image_adder_port'],
                                      'host':config['pyro']['image_adder_host']},
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
<<<<<<< HEAD
        callback_uri = 'PYRO:{}@{}:{}'.format(config['pyro']['image_adder_uri'],
                                               hostname,
                                               config['pyro']['image_adder_port'])

        command = '{} /home/jlaura/autocnet_server/bin/extract_features.py {} {}'
        command = command.format(config['python']['pybin'], path, callback_uri)
        if config['cluster']['cluster_log_dir'] is not None:
            log_out = config['cluster']['cluster_log_dir'] + '/%j.log'
=======

        callback_uri = 'PYRO:{}@{}:{}'.format(config['pyro']['image_adder_uri'],
                                                                       hostname,
                                                                       config['pyro']['image_adder_port'])

        command = '{} /home/acpaquette/repos/autocnet_server/bin/extract_features.py {} {}'
        command = command.format(config['python']['pybin'], path, callback_uri)
        if config['cluster']['cluster_log_dir'] is not None:
            log_out = config['cluster']['cluster_log_dir']  + '/%j.log'
>>>>>>> dd25bc9ce3d488b2e515e75b35c2a8b2e8f58714
        else:
            out = '%j.log'

        # Spawn the job and update the submission tracker
        res = spawn(command, name='AC_Extract', out=log_out, mem=config['cluster']['extractor_memory'])
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
            if self.job_status[d['path']]['count'] <= config['cluster']['maxfailures']:
                self.extract(d['path'], force=True)
