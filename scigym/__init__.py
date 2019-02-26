import distutils.version
import os
import sys
import warnings

from gym import error
from gym.utils import reraise

from gym.core import Env, GoalEnv, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper 
from gym.spaces.space import Space
from gym.envs import make, spec
from gym import logger

# import and register all scigym environments
import scigym.envs
from scigym.version import VERSION as __version__

# import all those gym classes/methods which we want to be able to call in scigym
__all__ = [
    # "Env", 
    # "Space", 
    # "Wrapper", 
    "make", 
    "spec"
    ]
