import datetime
import json
import csmapi

import requests

from plio.utils.utils import find_in_dict
from plio.io.io_json import NumpyEncoder


def data_from_cube(header):
    """
    Take an ISIS Cube header and normalize back to PVL keywords.
    """
    data = {}
    data['START_TIME'] = find_in_dict(header, 'StartTime')
    data['SPACECRAFT_NAME'] = find_in_dict(header, 'SpacecraftName')
    data['INSTRUMENT_NAME'] = find_in_dict(header, 'InstrumentId')
    data['SAMPLING_FACTOR'] = find_in_dict(header, 'SpatialSumming')
    data['SAMPLE_FIRST_PIXEL'] = find_in_dict(header, 'SampleFirstPixel')
    data['IMAGE'] = {}
    data['IMAGE']['LINES'] = find_in_dict(header, 'Lines')
    data['IMAGE']['SAMPLES'] = find_in_dict(header, 'Samples')
    data['TARGET_NAME'] = find_in_dict(header, 'TargetName')
    data['LINE_EXPOSURE_DURATION'] = find_in_dict(header, 'LineExposureDuration')
    data['SPACECRAFT_CLOCK_START_COUNT'] = find_in_dict(header, 'SpacecraftClockCount')
    return data

def create_camera(obj, url='http://pfeffer.wr.usgs.gov/v1/pds/',
                 plugin_name='USGS_ASTRO_LINE_SCANNER_PLUGIN',
                 model_name='USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL'):
    
    data = json.dumps(data_from_cube(obj.metadata), cls=NumpyEncoder)
    r = requests.post(url, data=data).json()

    # Get the ISD back and instantiate a local ISD for the image
    isd = csmapi.Isd.loads(r)

    # Create the plugin and camera as usual
    plugin = csmapi.Plugin().findPlugin(plugin_name)
    if plugin.canModelBeConstructedFromISD(isd, model_name):
        return plugin.constructModelFromISD(isd, model_name)
