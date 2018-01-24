from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def db_connect(p):
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(p['username'], p['password'], p['host'], p['port'], p['database']))
    Session = sessionmaker(bind=engine)
    return Session()
