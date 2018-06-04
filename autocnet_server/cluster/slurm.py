import subprocess
import textwrap

#TODO: slurm scripts to config

slurm_script = textwrap.dedent("""\
                              #!/bin/bash -l
                              #SBATCH -n 1
                              #SBATCH --mem-per-cpu {}
                              #SBATCH -J {}
                              #SBATCH -t {}
                              #SBATCH -o {}
                              #SBATCH -p {}
                              #SBATCH --exclude=neb[13-20],gpu1
                              {}""")

slurm_array = textwrap.dedent("""\
                              #!/bin/bash -l
                              #SBATCH -n 1
                              #SBATCH --mem-per-cpu {}
                              #SBATCH -o {}
                              #SBATCH -J {}
                              #SBATCH -t {}
                              #SBATCH -p {}
                              #SBATCH --exclude=neb[13-20],gpu1
                              {} {}
""")

def spawn(command, name='AutoCNet', time='01:00:00', outdir='/home/jlaura/autocnet_server/%j.log', mem=2048, queue='shortall'):
    """
    f : str
        file path
    """

    process = subprocess.Popen(['sbatch'], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE)
    job_string = slurm_script.format(mem, name, time, outdir, queue, command)
    process.stdin.write(str.encode(job_string))
    out, err = process.communicate()
    if err:
        print('ERROR: ', err)
        return False

    # If the job log has the %j character, replace it with the actual job id
    #try:
    #job_id = [int(s) for s in out.split() if s.isdigit()][0]
    #job_string = job_string.replace('%j', '{}'.format(job_id))
    #except:
    #    pass
    return job_string

def spawn_jobarr(py, command, njobs, name='AutoCNet', time='01:00:00',mem=2048, queue='shortall', outdir=r"slurm-%A_%a.out"):
    
    process = subprocess.Popen(['sbatch', '--array', '1-{}'.format(njobs)],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)

    job_string = slurm_array.format(mem, outdir, name, time, queue, py, command)
    process.stdin.write(str.encode(job_string))
    out, err = process.communicate()
    if err:
        print('ERROR: ', err)
        return False
    return job_string
