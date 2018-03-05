import subprocess
import textwrap

slurm_script = textwrap.dedent("""\
                              #!/bin/bash -l
                              #SBATCH -n 1
                              #SBATCH --mem-per-cpu 8192
                              #SBATCH -J {}
                              #SBATCH -t {}
                              #SBATCH -o {}.log
                              #SBATCH -e {}.log
                              #SBATCH -p shortall
                              {}""")

def spawn(command, name='AutoCNet', time='01:00:00', out='out', error='error'):
    """
    f : str
        file path
    """

    process = subprocess.Popen(['sbatch'], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE)

    job_string = slurm_script.format(name, time, out, error, command)
    process.stdin.write(str.encode(job_string))

    out, err = process.communicate()
    if err:
        print(err)
        return False
    return True
