from csmapi import csmapi

# Register the usgscam plugin with the csmapi
from ctypes.util import find_library
import ctypes

lib = ctypes.CDLL(find_library('usgscsm'))
