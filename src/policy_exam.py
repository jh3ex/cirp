import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger


import json

import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot



from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np



class ExamRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)






def run_sequential(args, logger):

    # Init runner so we can get env info
    # Send all args to runner
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    # Runner should return environment information
    env_info = runner.get_env_info()
    print(env_info)
    # Number of agents
    args.n_agents = env_info["n_agents"]
    # Number of actions
    args.n_actions = env_info["n_actions"]
    # The shape of actions
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    # Scheme is how we organize data in the replay buffer
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    print(scheme)
    groups = {
        "agents": args.n_agents
    }

    # This means we only preprocess actions data
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    # We can also add preprocess on stata. obs, etc.

    # Setup the replay buffer
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()



    logger.console_logger.info("Loading model from {}".format(model_path))
    learner.load_models(model_path)
    runner.t_env = timestep_to_load



    logger.console_logger.info("Beginning examing for {} timesteps".format(args.t_max))

    """
    RUNNING occurs here
    """
    # Run for **a whole episode** at a time
    runner.run(test_mode=True)


    runner.close_env()
    logger.console_logger.info("Finished Examing")




def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    # args: {"sacred": int, "model_path": path string, "timestep": int}

    with open(os.path.join(os.path.dirname(__file__), "sacred", params["sacred"], "config.json"), "r") as fname:
        config_dict = json.load(fname)

    config_dict["checkpoint_path"] = params["model_path"]
    config_dict["timestep_to_load"] = params["timestep"]

    config_dict["model_path"] = os.path.join(args["checkpoint_path"], args["timestep_to_load"])

    config = config_copy(config_dict)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    args = SN(**config)

    SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
    logger = get_logger()
    results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


#    config_dict = recursive_dict_update(config_dict, sacred_config)

#    env_config = _get_config(params, "--env-config", "envs")
#    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
#    config_dict = recursive_dict_update(config_dict, env_config)
#    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    # ex.add_config(config_dict)

    # Save to disk by default for sacred
#    logger.info("Saving to FileStorageObserver in results/sacred.")
    # file_obs_path = os.path.join(results_path, "sacred")
    # ex.observers.append(FileStorageObserver.create(file_obs_path))

    # ex.run_commandline(params)
