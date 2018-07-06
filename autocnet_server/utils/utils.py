import datetime
import os

def create_output_path(ds, outdir=None):
    image_name = os.path.basename(ds.file_name)
    image_path = os.path.dirname(ds.file_name)

    if outdir is None:
        outh5 = os.path.join(image_path, image_name + '_kps.h5')
    else:
        outh5 = os.path.join(outdir, image_name + '_kps.h5')

    return outh5

def slurm_walltime_to_seconds(walltime):
    """
    Convert a slurm defined walltime in the form
    HH:MM:SS into a number of seconds.

    Parameters
    ----------
    walltime : str
               In the form HH:MM:SS

    Returns
    -------
    d : int
        The number of seconds the walltime represents

    Examples
    >> walltime = '01:00:00'
    >> sec = slurm_walltime_to_seconds(walltime)
    >> sec
    3600
    """
    walltime = walltime.split(':')
    walltime = list(map(int,walltime))
    d = datetime.timedelta(hours=walltime[0],
                       minutes=walltime[1],
                       seconds=walltime[2])

    return d.seconds