import os
import yaml

#Load the config file
with open(os.environ['autocnet_config'], 'r') as f:
        config = yaml.load(f)