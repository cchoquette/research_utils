import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('expid', type=str, help='name of experiment')
parser.add_argument('--slurm_id', required=True, type=str, help='slurm job id for checkpointing')

