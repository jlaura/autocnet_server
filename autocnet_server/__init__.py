import os
import warnings
import yaml

#Load the config file
try:
    with open(os.environ['autocnet_config'], 'r') as f:
        config = yaml.load(f)
except:
    warnings.warn('No autocnet_config environment variable set. Defaulting to an en empty configuration.')
    config = {}