import autocnet
import argparse
import glob

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from flask import Flask, request, jsonify

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autocnet_server.db.model import Images, Keypoints, Base

app = Flask(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('db', help='Name of the database')
    parser.add_argument('-u', '--username', help='The username for the database')
    parser.add_argument('-p', '--password', help='The database password')
    parser.add_argument('-l', '--host', help='The database host')
    parser.add_argument('-o', '--port', help='The datbase port')
    return parser.parse_args()

def db_connect(p):
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(p.username, p.password, p.host, p.port, p.db))
    Session = sessionmaker(bind=engine)
    return Session()

@app.route('/')
def home():
    return 'Endpoints are "imgages" or "keypoints"'

@app.route('/images')
def image():
    res = session.query(Images).all()
    return jsonify([r.__repr__() for r in res])

@app.route('/images/add')
def add_image():
    name = request.args.get('name')
    path = request.args.get('path')
    fp = request.args.get('footprint')
    i = Images(name=name, path=path, footprint=fp)
    session.add(i)
    session.commit()
    return i.__repr__()

@app.route('/images/<id>')
def get_image(id):
    res = session.query(Images).filter(Images.id == id).first()
    return res.__repr__()

@app.route('/images/<id>/delete')
def delete_image(id):
    session.query(Images).filter(Images.id==id).delete()
    session.commit()
    return True

@app.route('/keypoints')
def keypoints():
    # Return a list of images that have keypoints
    pass

@app.route('/keypoints/add')
def add_keypoints():
    id = request.args.get('id')
    convex_hull = request.args.get('hull')
    path = request.args.get('path')
    nkeypoints = request.args.get('nkeypoints')

    k = Keypoints(image_id=id, convex_hull=convex_hull,
                  path=path, nkeypoints=nkeypoints)
    session.add(k)
    session.commit(k)
    return jsonify(k.__repr__)

@app.route('/keypoints/<id>')
def get_keypoints(id):
    res = session.query(Keypoints).filter(Keypoints.id == id).first()
    return res.__repr__()

@app.route('/extract/many')
def extract_many():
    path = request.args.get('path')
    extension = request.args.get('extension')
    search_dir = os.path.join(path, '*.{}'.format(extension))
    files = glob.glob(search_dir)

    for f in files:
        spawn_extraction(f)

    # The return here should be a join of the data that has been extracted.
    return jsonify(files)

@app.route('/extract/one')
def extract_one():
    path = request.args.get('path')
    spawn_extraction(path)
    return 'Success'

if __name__ == '__main__':
    args = parse_args()
    session = db_connect(args)

    app.run(debug=True)
