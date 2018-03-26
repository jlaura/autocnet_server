### Python Configuration ###
# Set the Python that should be used for cluster jobs
pyroot = '/home/jlaura/anaconda3/envs/ct/'
pybin = pyroot + 'bin/python' # Do not change me


### Cluster Configuration ###
# The number of times to retry a failing cluster job
maxfailures = 3
# Location to put <jobid.log> files for cluster jobs
cluster_log_dir = '/home/jlaura/logs'
cluster_submission = 'slurm'  # or `pbs`
tmp_scratch_dir = '/scratch/jlaura'

# The amount of RAM (in MB) to request for jobs
extractor_memory = 8192
processing_memory = 4000

### Database Configuration ###
database_username = 'jay'
database_password = 'abcde'
database_host = 'smalls'
database_port = 8001
# The name of the database to connect to.  Tables will be created inside this DB.
database_name = 'test2'
# The number of seconds to wait while attemping to connect to the DB.
timeout = 500

### Image Adder Configuration ###
image_adder_uri = 'ia'
image_adder_port = 8005
image_adder_host = '0.0.0.0'

### Candidate Graph Configuration ###
candidate_graph_uri = 'ncg'
candidate_graph_port = 8004
candidate_graph_host = '0.0.0.0'

### Spatial Reference Setup ###
srid = 949900
semimajor_rad = 3396190  # in meters
semiminor_rad = 3376200  # in meters
proj4_str = '+proj=longlat +a=3396190 +b=3376200 +no_defs'

### Working Directories ###
vrt_dir = '/scratch/jlaura/ctx/vrt'
