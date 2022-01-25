from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env
from envs.multiagentenv import MultiAgentEnv
# from envs.smart_man_sim.smart_man import SmartEnv
from envs.smart_man_sim.smart_man_flat import SmartEnv
import sys
import os

from envs.production_system.production_discrete import production_discrete
from envs.production_system.production_hmm import production_hmm

"""
This script registers the multi-agents environment that we need to to test on
"""
def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["smart_man"] = partial(env_fn, env=SmartEnv)
REGISTRY["smart_man_flat"] = partial(env_fn, env=SmartEnv)
REGISTRY["production_discrete"] = partial(env_fn, env=production_discrete)
REGISTRY["production_hmm"] = partial(env_fn, env=production_hmm)

#TODO I need to register my environment here
# REGISTRY["sman"] = partial(env_fn, env=StarCraft2Env)


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
