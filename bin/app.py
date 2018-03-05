import autocnet
import argparse
from collections import defaultdict
import glob
import json

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from flask import Flask, request, jsonify

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, aliased
from geoalchemy2.elements import WKTElement

from shapely.geometry import shape
from geoalchemy2.shape import from_shape

from cluster_spawn import spawn

from autocnet_server.db.model import Images, Keypoints, Base
from autocnet_server.graph.graph import NetworkCandidateGraph

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

@app.route('/images/post', methods=['GET', 'POST'])
def post_image():
    if request.method == 'POST':
        d = request.form

        footprint_ll = d['footprint_latlon']
        if footprint_ll:
            gj = json.loads(footprint_ll)
            fp_shapely = shape(gj)
            footprint_ll = WKTElement(fp_shapely.wkt, srid=949900)

        footprint_bf = d['footprint_bodyfixed']
        if footprint_bf:
            footprint_bf = WKTElement(footprint_bf)

        i = Images(name=d['name'], path=d['path'],
                       footprint_latlon=footprint_ll)
        session.add(i)
        session.commit()
        return json.dumps({"id":i.id})
    elif request.method == 'GET':
        # TODO: Write the post requirements
        print('Signature')

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

@app.route('/keypoints/post', methods=['GET', 'POST'])
def post_keypoints():
    if request.method == 'POST':
        d = request.form
        # Write the keypoints to the keypoints db table
        k = Keypoints(image_id=d['id'], path=d['outpath'],
                      nkeypoints=d['nkeypoints'])
        session.add(k)
        session.commit()
        return 'Success'

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

def extract(path):
    res = session.query(Images).filter(Images.path == path).first()
    if res:
        return 'Image already processed'

    command = '/home/jlaura/anaconda3/envs/ct/bin/python /home/jlaura/autocnet_server/bin/extract_features.py {}'
    command = command.format(path)
    res = spawn(command)
    return res

@app.route('/extract/many')
def extract_many():
    path = request.args.get('path')
    extension = request.args.get('extension')
    search_dir = os.path.join(path, '*.{}'.format(extension))
    files = glob.glob(search_dir)
    results = []
    for f in files:
        res = extract(f)
        results.append(res)
    return jsonify(files)

@app.route('/extract/one')
def extract_one():
    path = request.args.get('path')
    res = extract(path)
    return 'Success'

@app.route('/match/ring', methods=['GET', 'POST'])
def match_ring():
    command = '/home/jlaura/anaconda3/envs/ct/bin/python /home/jlaura/autocnet_server/bin/ring_match.py'
    res = spawn(command)
    return res

@app.route('/graph')
def graph():
    return 'GRAPH'

@app.route('/graph/create')
def graph_create():
    image_alias = aliased(Images)
    query = session.query(Images, image_alias)
    res = query.filter(Images.id < image_alias.id, Images.id != image_alias.id,
                       Images.footprint_latlon.intersects(image_alias.footprint_latlon))

    adjacency = defaultdict(list)
    adjacency_lookup = {}
    for r in res:
        s = r[0]
        d = r[1]
        adjacency_lookup[s.path] = s.id
        adjacency_lookup[d.path] = d.id
        if s.path != d.path:
            adjacency[s.path].append(d.path)

    data = {'adjacency':adjacency,
            'lookup':adjacency_lookup}
    cg = NetworkCandidateGraph.from_adjacency(adjacency,
                                              node_id_map=adjacency_lookup)
    return jsonify(data)



if __name__ == '__main__':
    args = parse_args()
    session = db_connect(args)
    # Host this way makes the dev server accessible remotely
    app.run(host='0.0.0.0', debug=True, port=8003)
