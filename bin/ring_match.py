import argparse
import json
import os
import sys
import time

import numpy as np
import ogr
import h5py
import pyproj

from autocnet.matcher.cpu_ring_matcher import ring_match, add_correspondences
from autocnet.io.keypoints import from_hdf

import hotqueue as hq

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ringradius', default=100)
    parser.add_argument('-m', '--maxradius', default=1200)
    parser.add_argument('-p', '--targetpoints', type=int, default=25)
    parser.add_argument('-t', '--tolerance', default=0.01)
    return parser.parse_args()

def match(msg, args):

    #try:
    # Load the npz file

    ref_kps, ref_desc = from_hdf(msg['spath'])
    tar_kps, tar_desc = from_hdf(msg['dpath'])

    # Default message
    data = {'success':False,
            'source': msg['sidx'], 'destin': msg['didx'],
            'callback':'ring_matcher_callback'}

    #TODO: Pull geom out - the ring matcher can handle all and singe we already
    # read all the kps, we are good to go.
    """    if 'geom' in msg.keys():
        fp = ogr.CreateGeometryFromWkt(msg['geom'])
        # Get the ref_kps that overlap
        ecef = pyproj.Proj(proj='geocent', a=3396190.0, b=3376200)
        lla = pyproj.Proj(proj='longlat', a=3396190, b=3376200)

        lons, lats, alts = pyproj.transform(ecef, lla, ref_kps.xm.values, ref_kps.ym.values, ref_kps.zm.values)
        ref_idx = []
        for i in range(len(lons)):
            g = ogr.Geometry(ogr.wkbPoint)
            g.AddPoint(lons[i], lats[i])
            if fp.Contains(g):
                ref_idx.append(i)

        lons, lats, alts = pyproj.transform(ecef, lla, tar_kps.xm.values, tar_kps.ym.values, tar_kps.zm.values)
        tar_idx = []
        for i in range(len(lons)):
            g = ogr.Geometry(ogr.wkbPoint)
            g.AddPoint(lons[i], lats[i])
            if fp.Contains(g):
                tar_idx.append(i)

        ref_kps = ref_kps.iloc[ref_idx]
        ref_desc = ref_desc[ref_idx]
        ref_index = sindex[ref_idx]
        tar_kps = tar_kps.iloc[tar_idx]
        tar_desc = tar_desc[tar_idx]
        tar_index = dindex[tar_idx]"""

    ref_feats = ref_kps[['x', 'y', 'xm', 'ym', 'zm']].values
    tar_feats = tar_kps[['x', 'y', 'xm', 'ym', 'zm']].values

    _, _, pidx, ring = ring_match(ref_feats, tar_feats,
                                  ref_desc, tar_desc,
                                  ring_radius=args.ringradius,
                                  max_radius=args.maxradius,
                                  target_points=args.targetpoints,
                                  tolerance_val=args.tolerance)


    if pidx is None:
        print('Unable to find a solution.')
        return data

    # Now densify the matches if a ring has been found
    print('Initial Pass Resulted in {} matches'.format(len(pidx)))
    print('Distance ring: {}'.format(ring))

    in_feats = ref_feats[pidx[:,0]][:,:2]  # all reference points[those selected by ring matcher][x,y coords]
    xextent = (np.min(in_feats[:,1]), np.max(in_feats[:,1]))
    yextent = (np.min(in_feats[:,0]), np.max(in_feats[:,0]))
    refs_to_add = add_correspondences(in_feats,
                                      ref_feats, tar_feats,
                                      ref_desc, tar_desc,
                                      xextent, yextent, ring,
                                      8, 8, target_points=15,
                                      search_radius=int(args.ringradius / 3),
                                      max_search_radius=args.ringradius)
    refs_to_add = [i for i in refs_to_add if len(i)]

    if refs_to_add:
        print('Adding {} correspondences'.format(len(refs_to_add)))
        stacked_refs_to_add = np.vstack(refs_to_add)
        pidx = np.vstack((pidx, stacked_refs_to_add))
        # Get the unique rows: https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array
        # Perform lex sort and get sorted data
        sorted_idx = np.lexsort(pidx.T)
        sorted_data =  pidx[sorted_idx,:]
        # Get unique row mask
        row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
        # Get unique rows
        pidx = sorted_data[row_mask]
    else:
        print('no additional references to add')

    # Check for duplicates
    l = pidx[:
    ,1].tolist()
    clean = [i for i, x in enumerate(l) if l.count(x) == 1]
    pidx = pidx[clean, :]

    # Convert from the found indices into the footprint indices
    #ref_idx = ref_feats[pidx[:,0]].index.values  # all_kps[kps_in_overlap][selected]
    #tar_idx = tar_feats[pidx[:,1]].index.values

    # Package the data to round trip to the server
    data['success'] = True
    data['sidx'] = pidx[:,0]
    data['didx'] = pidx[:,1]
    data['ring'] = ring

    print(data)

    return data

def finalize(data, queue):
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            data[k] = v.tolist()

    queue.put(data)


if __name__ == '__main__':
    args = parse_args()
    queue = hq.HotQueue('processor', serializer=json, host="smalls", port=8000, db=0)
    fqueue = hq.HotQueue('completed', serializer=json, host="smalls", port=8000, db=0)
    msg = queue.get()
    data = match(msg, args)
    finalize(data, fqueue)
