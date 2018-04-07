import argparse
import sys
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database
from autocnet_server.db.model import Images, Keypoints, Base
from autocnet_server import config

# TODO Remove if necessary because we no longer use it
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('db', help='Name of the database')
    parser.add_argument('-u', '--username', help='The username for the database')
    parser.add_argument('-p', '--password', help='The database password')
    parser.add_argument('-l', '--host', help='The database host')
    parser.add_argument('-o', '--port', help='The datbase port')
    parser.add_argument('-c', '--config', help='An optional database config file that defines the above flags')
    return vars(parser.parse_args())

def create_db():
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(config['database_creation']['admin'],
                                                                config['database_creation']['password'],
                                                                config['database']['database_host'],
                                                                config['database']['database_port'],
                                                                config['database']['database_name']))
    if not database_exists(engine.url):
        create_database(engine.url)
    # Enable postgis
    connection = engine.connect()

    result = connection.execute('CREATE EXTENSION postgis')
    result = connection.execute("""INSERT into spatial_ref_sys (srid, auth_name, auth_srid, proj4text, srtext) values ( 949900, 'iau2000', 49900, '+proj=longlat +a=3396190 +b=3376200 +no_defs ', 'GEOGCS["Mars 2000",DATUM["D_Mars_2000_IAU_IAG",3396190.0,169.89444722361179]],PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]]');""")
    result = connection.execute("""ALTER DATABASE {} OWNER TO {}""".format(config['database']['database_name'],config['database']['database_username']))

    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(config['database']['database_username'],
                                                                config['database']['database_password'],
                                                                config['database']['database_host'],
                                                                config['database']['database_port'],
                                                                config['database']['database_name']))

    # Create the tables
    Base.metadata.bind = engine
    Base.metadata.create_all(engine)

if __name__ == '__main__':
    create_db()
