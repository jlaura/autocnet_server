import argparse
import os
import sys

# Get the server installing
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd

from autocnet.matcher.cuda_extractor import extract_features
from autocnet.utils.utils import tile
from autocnet.io.keypoints import to_hdf
from autocnet.camera.csm_camera import create_camera
from autocnet.cg import footprint
from autocnet_server.db.model import Images, Keypoints

from autocnet_server.utils.utils import create_output_path
from autocnet_server.db.connection import db_connect

from geoalchemy2.elements import WKTElement

import json

from plio.io.io_gdal import GeoDataset
import pyproj
import ogr

import autocnet
funcs = {'vlfeat':autocnet.matcher.cpu_extractor.extract_features}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument('-t', '--threshold', help='The threshold difference between DN values')
    parser.add_argument('-n', '--nfeatures', help='The number of features to extract. Default is max_image_dimension / 1.25', type=float)
    parser.add_argument('-m', '--maxsize',type=float, default=6e7, help='The maximum number of pixels before tiling is used to extract keypoints.  Default: 6e7')
    parser.add_argument('-e', '--extractor', default='vlfeat', choices=['cuda', 'vlfeat'], help='The extractor to use to get keypoints.')
    parser.add_argument('-c', '--camera', action='store_false', help='Whether or not to compute keypoints coordinates in body fixed as well as image space.')
    parser.add_argument('-o', '--outdir', type=str, help='The output directory')
    parser.add_argument('-d','--database', help='The database configuration definition.')
    return vars(parser.parse_args())

def extract(ds, extractor, maxsize):
    #TODO: Move this into a testable place
    if ds.raster_size[0] * ds.raster_size[1] > maxsize:
        slices = tile(ds.raster_size, tilesize=12000, overlap=250)
    else:
        slices = [[0,0,ds.raster_size[0], ds.raster_size[1]]]

    extractor_params = {'compute_descriptor': True,
                        'float_descriptors': True,
                        'edge_thresh':2.5,
                        'peak_thresh': 0.0001,
                        'verbose': False}
    keypoints = pd.DataFrame()
    descriptors = None
    for s in slices:
        xystart = [s[0], s[1]]
        array = ds.read_array(pixels=s)

        kps, desc = funcs[extractor](array, extractor_method='vlfeat', extractor_parameters=extractor_params)

        kps['x'] += xystart[0]
        kps['y'] += xystart[1]

        count = len(keypoints)
        keypoints = pd.concat((keypoints, kps))
        descriptor_mask = keypoints.duplicated()

        # Removed duplicated and re-index the merged keypoints
        keypoints.drop_duplicates(inplace=True)
        keypoints.reset_index(inplace=True, drop=True)

        if descriptors is not None:
            descriptors = np.concatenate((descriptors, desc))
        else:
            descriptors = desc
        descriptors = descriptors[~descriptor_mask]
        #self.descriptors = descriptors

    return keypoints, descriptors

if __name__ == '__main__':
    #TODO: Tons of logic in here to get extracted

    # Setup the metadata obj that will be written to the db
    metadata = {}

    # Parse args and grab the file handle to the image
    kwargs = parse_args()
    input_file = kwargs.pop('input_file', None)
    ds = GeoDataset(input_file)

    # Check to see if the image is already in the database
    # Create a database session to use
    with open(kwargs['database'], 'r') as f:
        dbparams = json.load(f)
    session = db_connect(dbparams)
    res = session.query(Images).filter(Images.name == ds.base_name).first()
    if res:
        print('Image already exists in database.')
        sys.exit()

    # Extract the correspondences
    extractor = kwargs.pop('extractor')
    maxsize = kwargs.pop('maxsize')
    keypoints, descriptors = extract(ds, extractor, maxsize)

    # Create a camera model for the image
    camera = kwargs.pop('camera')
    if camera:
        camera = create_camera(ds)

        # Project the sift keypoints to the ground
        def func(row, args):
            camera = args[0]
            gnd = getattr(camera, 'imageToGround')(row[1], row[0], 0)
            return gnd

        feats = keypoints[['x', 'y']].values
        gnd = np.apply_along_axis(func, 1, feats, args=(camera, ))
        gnd = pd.DataFrame(gnd, columns=['xm', 'ym', 'zm'], index=keypoints.index)
        keypoints = pd.concat([keypoints, gnd], axis=1)

    # Write the correspondences to disk
    outdir = kwargs.pop('outdir')
    outpath = create_output_path(ds, outdir)
    to_hdf(keypoints, descriptors, outpath)

    # Compute the image footprint
    if camera:
        footprint_latlon = footprint.generate_latlon_footprint(camera).ExportToWkt()
        footprint_bodyfixed = footprint.generate_bodyfixed_footprint(camera).ExportToWkt()
    else:
        footprint_latlon = None
        footprint_bodyfixed = None

    # Write the image to the images db table
    i = Images(name=ds.base_name, path=ds.file_name, footprint_latlon=footprint_latlon, srid=949900),
               footprint_bodyfixed=footprint_bodyfixed)
    session.add(i)
    session.commit()

    # Write the keypoints to the keypoints db table
    k = Keypoints(image_id=i.id, path=outpath, nkeypoints=len(keypoints))
    session.add(k)
    session.commit()
    session.close()
