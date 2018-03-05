import json

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float, ForeignKey, Boolean, LargeBinary
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship, backref
from geoalchemy2 import Geometry
from geoalchemy2.shape import to_shape


Base = declarative_base()

attr_dict = {'__tablename__':None,
             '__table_args__': {'useexisting':True},
             'id':Column(Integer, primary_key=True, autoincrement=True),
             'name':Column(String),
             'path':Column(String),
             'footprint':Column(Geometry('POLYGON')),
             'keypoint_path':Column(String),
             'nkeypoints':Column(Integer),
             'kp_min_x':Column(Float),
             'kp_max_x':Column(Float),
             'kp_min_y':Column(Float),
             'kp_max_y':Column(Float)}

def create_table_cls(name, clsname):
    attrs = attr_dict
    attrs['__tablename__'] = name
    return type(clsname, (Base,), attrs)

class Keypoints(Base):
    __tablename__ = 'keypoints'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"))
    convex_hull_image = Column(Geometry('POLYGON'))
    convex_hull_latlon = Column(Geometry('POLYGON', srid=949900, dimension=3))
    path = Column(String)
    nkeypoints = Column(Integer)

    def __repr__(self):
        try:
            chll = to_shape(self.convex_hull_latlon).__geo_interface__
        except:
            chll = None
        return json.dumps({'id':self.id,
                           'image_id':self.image_id,
                           'convex_hull':self.convex_hull_image,
                           'convex_hull_latlon':chll,
                           'path':self.path,
                           'nkeypoints':self.nkeypoints})

class Matches(Base):
    __tablename__ = 'matches'
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(Integer, nullable=False)
    source_idx = Column(Integer, nullable=False)
    destination = Column(Integer, nullable=False)
    destination_idx = Column(Integer, nullable=False)
    lat = Column(Float)
    lon = Column(Float)
    geom = Column(Geometry('POINTZ', dimension=3, srid=949900, spatial_index=True))
    source_x = Column(Float)
    source_y = Column(Float)
    destination_x = Column(Float)
    destination_y = Column(Float)


class Cameras(Base):
    __tablename__ = 'cameras'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"))
    camera = Column(LargeBinary)


class Images(Base):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    path = Column(String)
    active = Column(Boolean)
    footprint_latlon = Column(Geometry('POLYGONZ', srid=949900, dimension=3, spatial_index=True))
    footprint_bodyfixed = Column(Geometry('POLYGONZ', dimension=3))
    #footprint_bodyfixed = Column(Geometry('POLYGON',dimension=3))

    # Relationships
    keypoints = relationship(Keypoints, passive_deletes='all', backref="images", uselist=False)
    cameras = relationship(Cameras, passive_deletes='all', backref='images', uselist=False)

    def __repr__(self):
        try:
            footprint = to_shape(self.footprint_latlon).__geo_interface__
        except:
            footprint = None
        return json.dumps({'id':self.id,
                'name':self.name,
                'path':self.path,
                'footprint_latlon':footprint,
                'footprint_bodyfixed':self.footprint_bodyfixed})
