import distutils.version
import os
import sys
import warnings

from gym import error

# from gym.core import Env, GoalEnv, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
# from gym.spaces import Space, Discrete, Box
from gym.envs import make, spec
from gym import logger

# import and register all scigym environments
import scigym.envs
from scigym.version import VERSION as __version__

# import all those gym classes/methods which we want to be able to call in scigym
__all__ = [
    # "Env", 
    # "Wrapper", 
    # "Space", 
    "make", 
    "spec",
    # "Discrete",
    # "Box"
    ]
