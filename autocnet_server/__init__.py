from pkg_resources import get_distribution, DistributionNotFound
import os.path
import socket

from sqlalchemy import create_engine, pool, orm

try:
    _dist = get_distribution('autocnet_server')
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, 'autocnet_server')):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Please install this project with setup.py'
else:
    __version__ = _dist.version


import os
import warnings
import yaml

#Load the config file and setup a global DB session factory
try:
    with open(os.environ['autocnet_config'], 'r') as f:
        config = yaml.load(f)
except:
    warnings.warn('No autocnet_config environment variable set. Defaulting to an en empty configuration.')
    config = {}

Session = None
if config:
    db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(config['database']['database_username'],
                                                  config['database']['database_password'],
                                                  config['database']['database_host'],
                                                  config['database']['pgbouncer_port'],
                                                  config['database']['database_name'])
    hostname = socket.gethostname()
    engine = create_engine(db_uri, poolclass=pool.NullPool,
                           connect_args={"application_name":"AutoCNet_{}".format(hostname)},
                           isolation_level="AUTOCOMMIT")                   
    Session = orm.session.sessionmaker(bind=engine) 
