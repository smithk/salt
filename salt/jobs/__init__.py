"""The :mod:`salt.jobs` subpackage provides interaction with distributed computing platforms."""

#from .jobs import LearningJobManager


#job_manager = LearningJobManager()
from .jobs import LocalJobManager, DistributedJobManager

__all__ = ['LocalJobManager', 'DistributedJobManager']
