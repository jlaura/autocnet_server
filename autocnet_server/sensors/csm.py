import datetime
import json
import os

from csmapi import csmapi
import jinja2
import requests

import numpy as np
import pyproj
import gdal
import ogr
import pvl

from plio.utils.utils import find_in_dict
from plio.io.io_json import NumpyEncoder


def data_from_cube(header):
    """
    Take an ISIS Cube header and normalize back to PVL keywords.
    """
    instrument_name = 'CONTEXT CAMERA'
    data = pvl.PVLModule([('START_TIME', find_in_dict(header, 'StartTime')),
                          ('SPACECRAFT_NAME', find_in_dict(header, 'SpacecraftName').upper()),
                          ('INSTRUMENT_NAME', instrument_name),
                          ('SAMPLING_FACTOR', find_in_dict(header, 'SpatialSumming')),
                          ('SAMPLE_FIRST_PIXEL', find_in_dict(header, 'SampleFirstPixel')),
                          ('TARGET_NAME', find_in_dict(header, 'TargetName').upper()),
                          ('LINE_EXPOSURE_DURATION', find_in_dict(header, 'LineExposureDuration')),
                          ('SPACECRAFT_CLOCK_START_COUNT', find_in_dict(header, 'SpacecraftClockCount')),
                          ('IMAGE', {'LINES':find_in_dict(header, 'Lines'),
                                    'LINE_SAMPLES':find_in_dict(header, 'Samples')})])

    return data

def create_camera(obj, url='http://pfeffer.wr.usgs.gov/v1/pds/',
                 plugin_name='USGS_ASTRO_LINE_SCANNER_PLUGIN',
                 model_name='USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'):
    data = data_from_cube(obj.metadata)

    data_serialized = {'label': pvl.dumps(data).decode()}
    r = requests.post(url, json=data_serialized).json()
    r['IKCODE'] = -1
    # Get the ISD back and instantiate a local ISD for the image
    isd = csmapi.Isd.loads(r)

    # Create the plugin and camera as usual
    plugin = csmapi.Plugin.findPlugin(plugin_name)
    if plugin.canModelBeConstructedFromISD(isd, model_name):
        return plugin.constructModelFromISD(isd, model_name)

def generate_gcps(camera, nnodes=5, semi_major=3396190, semi_minor=3376200):
    ecef = pyproj.Proj(proj='geocent', a=semi_major, b=semi_minor)
    lla = pyproj.Proj(proj='latlon', a=semi_major, b=semi_minor)

    isize = camera.getImageSize()
    isize = [isize.samp, isize.line]
    x = np.linspace(0,isize[1], 10)
    y = np.linspace(0,isize[0], 10)
    boundary = [(i,0.) for i in x] + [(isize[1], i) for i in y[1:]] +\
               [(i, isize[0]) for i in x[::-1][1:]] + [(0.,i) for i in y[::-1][1:]]
    gnds = np.empty((len(boundary), 3))
    for i, b in enumerate(boundary):
        gnd = camera.imageToGround(csmapi.ImageCoord(*b), 0)
        gnds[i] = [gnd.x, gnd.y, gnd.z]
    lons, lats, alts = pyproj.transform(ecef, lla, gnds[:,0], gnds[:,1], gnds[:,2])
    lla = np.vstack((lons, lats, alts)).T

    tr = zip(boundary, lla)

    gcps = []
    for i, t in enumerate(tr):
        l = '<GCP Id="{}" Info="{}" Pixel="{}" Line="{}" X="{}" Y="{}" Z="{}" />'.format(i, i, t[0][1], t[0][0], t[1][0], t[1][1], t[1][2])
        gcps.append(l)

    return gcps

def generate_latlon_footprint(camera, nnodes=5, semi_major=3396190, semi_minor=3376200):
    ecef = pyproj.Proj(proj='geocent', a=semi_major, b=semi_minor)
    lla = pyproj.Proj(proj='latlon', a=semi_major, b=semi_minor)

    isize = camera.getImageSize()
    isize = [isize.samp, isize.line]
    x = np.linspace(0,isize[1], 10)
    y = np.linspace(0,isize[0], 10)
    multipoly = ogr.Geometry(ogr.wkbMultiPolygon)
    boundary = [(i,0.) for i in x] + [(isize[1], i) for i in y[1:]] +\
               [(i, isize[0]) for i in x[::-1][1:]] + [(0.,i) for i in y[::-1][1:]]
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for i in boundary:
        gnd = camera.imageToGround(csmapi.ImageCoord(*i), 0)
        lons, lats, alts = pyproj.transform(ecef, lla, gnd.x, gnd.y, gnd.z)
        ring.AddPoint(lons, lats)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    multipoly.AddGeometry(poly)
    return multipoly

def generate_bodyfixed_footprint(camera, nnodes=5):
    isize = camera.imagesize[::-1]
    x = np.linspace(0,isize[1], 10)
    y = np.linspace(0,isize[0], 10)
    boundary = [(i,0.) for i in x] + [(isize[1], i) for i in y[1:]] +\
               [(i, isize[0]) for i in x[::-1][1:]] + [(0.,i) for i in y[::-1][1:]]
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for i in boundary:
        gnd = camera.imageToGround(*i, 0)
        ring.AddPoint(gnd[0], gnd[1], gnd[2])
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly

def generate_vrt(camera, raster_size, fpath, outpath=None, no_data_value=0):
    gcps = generate_gcps(camera)
    xsize, ysize = raster_size

    if outpath is None:
        outpath = os.path.dirname(fpath)
    outname = os.path.splitext(os.path.basename(fpath))[0] + '.vrt'
    outname = os.path.join(outpath, outname)

    xsize, ysize = raster_size
    vrt = r'''<VRTDataset rasterXSize="{{ xsize }}" rasterYSize="{{ ysize }}">
     <Metadata/>
     <GCPList Projection="{{ proj }}">
     {% for gcp in gcps -%}
       {{gcp}}
     {% endfor -%}
    </GCPList>
     <VRTRasterBand dataType="Float32" band="1">
       <NoDataValue>{{ no_data_value }}</NoDataValue>
       <Metadata/>
       <ColorInterp>Gray</ColorInterp>
       <SimpleSource>
         <SourceFilename relativeToVRT="0">{{ fpath }}</SourceFilename>
         <SourceBand>1</SourceBand>
         <SourceProperties rasterXSize="{{ xsize }}" rasterYSize="{{ ysize }}"
    DataType="Float32" BlockXSize="512" BlockYSize="512"/>
         <SrcRect xOff="0" yOff="0" xSize="{{ xsize }}" ySize="{{ ysize }}"/>
         <DstRect xOff="0" yOff="0" xSize="{{ xsize }}" ySize="{{ ysize }}"/>
       </SimpleSource>
     </VRTRasterBand>
    </VRTDataset>'''

    context = {'xsize':xsize, 'ysize':ysize,
               'gcps':gcps,
               'proj':'+proj=longlat +a=3396190 +b=3376200 +no_defs',
               'fpath':fpath,
               'no_data_value':no_data_value}
    template = jinja2.Template(vrt)
    tmp = template.render(context)
    warp_options = gdal.WarpOptions(format='VRT', dstNodata=0)
    gdal.Warp(outname, tmp, options=warp_options)
