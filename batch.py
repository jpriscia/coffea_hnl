import parsl
import os
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config

from parsl.providers import LocalProvider,CondorProvider,SlurmProvider
from parsl.channels import LocalChannel,SSHChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher

from parsl.addresses import address_by_hostname

def configure(memory=2048, nprocs=8, nodes=15):
    '''Configure the parsl scheduler (is it the right name?)
    arguments: 
      * memory: amount of memory per node (default: 2GB)
      * nprocs: number of cores per node (default: 16)
      * nodes: number of nodes (default: 20)
    '''
    wrk_init = f'''
    export XRD_RUNFORKHANDLER=1
    export X509_USER_PROXY={os.environ['X509_USER_PROXY']}
    '''
    
    sched_opts = f'''
    #SBATCH --cpus-per-task={nprocs}
    #SBATCH --mem-per-cpu={memory}
    '''
    
    
    slurm_htex = Config(
        executors=[
            HighThroughputExecutor(
                label="coffea_parsl_slurm",
                address=address_by_hostname(),
                prefetch_capacity=0,  
                max_workers=nprocs,
                provider=SlurmProvider(
                    channel=LocalChannel(),
                    launcher=SrunLauncher(),
                    init_blocks=nodes,
                    max_blocks=nodes*2,
                    nodes_per_block=1,
                    partition='all',
                    scheduler_options=sched_opts,   # Enter scheduler_options if needed
                    worker_init=wrk_init,         # Enter worker_init if needed
                    walltime='00:30:00'
                ),
            )
        ],
        #retries=3,
        strategy=None,
    )
    
    # parsl.set_stream_logger() # <-- log everything to stdout, WAAAAY too much
    
    return parsl.load(slurm_htex)
