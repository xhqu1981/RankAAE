#!/usr/bin/env python

import argparse
import socket

import torch
from rankaae.models.trainer import Trainer
from rankaae.utils.parameter import Parameters
from rankaae.utils.logger import create_logger
import os
import ipyparallel as ipp
import logging
import signal
import time
import numpy as np

class RankAAEMPIEngineSetLauncher(ipp.cluster.launcher.MPILauncher, ipp.cluster.launcher.EngineLauncher):
    @property
    def program(self):
        return self.engine_cmd

def get_parallel_map_func(rc):
    with rc[:].sync_imports():
        import torch
        from rankaae.models.trainer import Trainer
        from rankaae.utils.parameter import Parameters
        from rankaae.utils.logger import create_logger
        import os
        import socket
        import logging
        import signal
        import time
    rc[:].push(dict(run_training=run_training, 
                    timeout_handler=timeout_handler),
                    block=True)
    par_map = rc.load_balanced_view().map_sync
    return par_map

def timeout_handler(signum, frame):
    raise Exception("Training Overtime!")


def run_training(
        job_number, 
        work_dir, 
        train_config,  
        verbose, 
        data_file, 
        timeout_hours=0,
        logger = logging.getLogger("training")):
    work_dir = f'{work_dir}/training/job_{job_number+1}'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    # Set up a logger to record general training information
    logger = create_logger(f"subtraining_{job_number+1}", os.path.join(work_dir, "messages.txt"))
    # Set up a logger to record losses against epochs during training 
    loss_logger = create_logger(f"losses_{job_number+1}", os.path.join(work_dir, "losses.csv"), simple_fmt=True)

    if torch.get_num_interop_threads() > 2:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
    
    local_id = int(os.environ.get("SLURM_LOCALID", -1))
    if local_id < 0:
        local_id = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", -1))
    if local_id < 0:
        logger.info(f"Unable to find local rank ID, set to zero\n")
        local_id = 0
    ngpus_per_node = torch.cuda.device_count()
    igpu = local_id % ngpus_per_node if torch.cuda.is_available() else -1
    
    start = time.time()
    logger.info(f"Training started for trial {job_number+1}.")

    trainer = Trainer.from_data(
        data_file,
        igpu = igpu,
        verbose = verbose,
        work_dir = work_dir,
        config_parameters = train_config,
        logger = logger,
        loss_logger = loss_logger,
    )
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_hours * 3600))

    metrics = trainer.train()
    logger.info(metrics)

    signal.alarm(0)
    
    time_used = time.time() - start
    logger.info(f"Training finished. Time used: {time_used:.2f}s.\n\n")
    
    return metrics, time_used


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config for training parameter in YAML format')
    parser.add_argument('-p', '--processes', type=int, default=1,
                        help='Number of processes')
    parser.add_argument('-w', "--work_dir", type=str, default='.',
                        help="Working directory to write the output files")
    args = parser.parse_args()

    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    train_config = Parameters.from_yaml(os.path.join(work_dir, args.config))
    assert os.path.exists(work_dir)

    verbose = train_config.get("verbose", False)
    trials = train_config.get("trials", 1)
    data_file = os.path.join(work_dir, train_config.get("data_file", None))
    timeout = train_config.get("timeout", 10)

    # Start Logger
    logger = create_logger("Main training:", f'{work_dir}/main_process_message.txt', append=True)
    logger.info("START")

    training_func_params = [
        list(range(trials)),
        [work_dir] * trials,
        [train_config] * trials,
        [verbose] * trials,
        [data_file] * trials,
        [timeout] * trials,
        [logger] * trials]
    
    start = time.time()
    if args.processes > 1:
        mpi_cmd = train_config.get("mpi_cmd", "srun")
        ipp.cluster.launcher.MPILauncher.mpi_cmd = [mpi_cmd]
        ip = socket.gethostbyname(socket.gethostname())
        with ipp.Cluster(engines=RankAAEMPIEngineSetLauncher, n=args.processes,
                         controller_ip='*', controller_location=ip, 
                         profile_dir=f'{work_dir}/ipypar') as rc:
            time, par_map = get_parallel_map_func(rc)
            nprocesses = len(rc.ids)
            assert nprocesses > 1
            logger.info("Running with {} processes.".format(nprocesses))
            result = par_map(run_training, *training_func_params)
    else:
        logger.info("Running with a single process.")
        result = map(run_training, *training_func_params)

    time_trials = np.array([r[1] for r in list(result)])
    logger.info(
        f"Time used for each trial: {time_trials.mean():.2f} +/- {time_trials.std():.2f}s.\n" + 
        ' '.join([f"{t:.2f}s" for t in time_trials])
    )
    
    end = time.time()
    logger.info(
        f"Total time used: {end-start:.2f}s for {trials} trails " +
        f"({(end-start)/trials:.2f} each on average)."
    )
    logger.info("END\n\n")


if __name__ == '__main__':
    main()
