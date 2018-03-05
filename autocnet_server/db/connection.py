from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import create_session, scoped_session, sessionmaker

@contextmanager
def db_connect(p):
    try:
        engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(p['username'], p['password'], p['host'], p['port'], p['database']))
        session_factory = sessionmaker(bind=engine)
        Session = scoped_session(lambda: create_session(bind_engine))
        yield Session()
    finally:
        Session.remove()
