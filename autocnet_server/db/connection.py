from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import create_session, scoped_session, sessionmaker

from autocnet_server import config

def new_connection():
    db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(config.database_username,
                                                  config.database_password,
                                                  config.database_host,
                                                  config.pgbouncer_port,
                                                  config.database_name)
    engine = sqlalchemy.create_engine(db_uri,
                                      poolclass=sqlalchemy.pool.NullPool)
    Session = sqlalchemy.orm.sessionmaker(bind=engine, autocommit=True)
    return Session()

@contextmanager
def db_connect(p):
    try:
        engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(p['username'], p['password'], p['host'], p['port'], p['database']))
        session_factory = sessionmaker(bind=engine)
        Session = scoped_session(lambda: create_session(bind_engine))
        yield Session()
    finally:
        Session.remove()
