import argparse

_parser = None


def get_parser():
  """Return Foreshadow logger instance.
  Will create and setup if needed, else will return the previously setup
  logger.
  Returns:
      foreshadow logger instance.
  """
  global _parser

  if _parser:
    return _parser

  if _parser is not None:
    return _parser
  _parser = argparse.ArgumentParser('')
  _parser.add_argument('expid', type=str, help='name of experiment')
  _parser.add_argument('--slurm_id', required=True, type=str, help='slurm job id for checkpointing')
  return _parser
