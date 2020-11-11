from .core import Simulation
from .scenes import get_map_params

from envs.multiagentenv import MultiAgentEnv
from absl import logging
import numpy as np
from itertools import product


class SmartEnv(MultiAgentEnv):
    def __init__(self,
                 map_name,
                 seed=None,
                 continuing_episode=False,
                 obs_all_health=True,
                 obs_own_health=True,
                 obs_all_cost=True,
                 obs_own_cost=True,
                 obs_last_action=False,
                 obs_profit=True,
                 obs_instead_of_state=False,
                 obs_timestep_number=False,
                 state_last_action=True,
                 state_own_cost=True,
                 state_profit=True,
                 state_timestep_number=False,
                 reward_scale=True,
                 reward_scale_rate=20,
                 replay_dir="",
                 replay_prefix="",
                 debug=False):

        """
        Create a SmartEnv environment.
        Parameters
        ----------
        map_name : str, optional
            The name of the smart manu map to play (default is "8m"). The full list
            can be found in scenes.py.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        continuing_episode : bool, optional
            Whether to consider episodes continuing or finished after time
            limit is reached (default is False).
        obs_all_health : bool, optional
            Agents receive the health of all units (in the sight range) as part
            of observations (default is True).
        obs_own_health : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_all_cost : bool, optional
            Agents receive the cost of all units (in the sight range) as part
            of observations (default is True).
        obs_own_cost : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_last_action : bool, optional
            Agents receive the last actions of all units (in the sight range)
            as part of observations (default is False).
        obs_instead_of_state : bool, optional
            Use combination of all agents' observations as the global state
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        state_profit :  bool, optional
            Whether the state include the current profit of the episode
            (default is True).
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).
        """
        # Map arguments
        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        #self.n_agents = map_params["n_agents"]
        self.n_agents = 1
        self.n_cells = map_params['n_cells']
        self.episode_limit = map_params["limit"]
        self.actions_output = map_params['actions']

        # Observations and states
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_own_cost = obs_own_cost
        self.obs_all_cost = obs_all_cost
        self.obs_profit = obs_profit
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_last_action = obs_last_action
        self.obs_timestep_number = obs_timestep_number
        self.state_last_action = state_last_action
        self.state_own_cost = state_own_cost
        self.state_profit = state_profit
        self.state_timestep_number = state_timestep_number

        # Rewards args
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate

        # Other
        self.continuing_episode = continuing_episode
        self._episode_count = 0
        self._seed = seed
        self.debug = debug
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix
        self.accum_profit = 0
        self.current_profit = 0

        # Actions
        self.n_actions = map_params["n_actions"] ** map_params["n_agents"]

        self.agents = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.last_action_idx = None
        self._init_action_map(map_params["n_agents"], map_params["n_actions"])


    def _init_action_map(self, n_agents, n_actions):
        action = list(np.arange(n_actions))
        actions = []
        for i in range(n_agents):
            actions.append(action)
        actions_list = list(product(*actions))

        idx2action = {}
        for i in range(len(actions_list)):
            idx2action[i] = actions_list[i]
        action2idx = {v:k for k, v in idx2action.items()}

        self.idx2action = idx2action
        self.action2idx = action2idx


    def _launch(self):
        """Launch the simulation"""
        self.sim = Simulation(self.map_name)

    def reset(self):
        self._episode_steps = 0
        self._launch()
        self.current_profit = 0
        map_params = get_map_params(self.map_name)
        n_agents = map_params["n_agents"]
        self.step([0] * n_agents)
        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))
        return self.get_obs(), self.get_state()

    def get_obs_agent(self):
        """Returns observation for all agents.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        nf_own = 9 #front buffer percentage, following buffer percentage, agent id,
                    #saturation, health state, state time, under_m, last_action, own_cost
        map_params = get_map_params(self.map_name)
        n_agents = map_params['n_agents']
        feats = np.zeros((n_agents, nf_own))
        actions = self.idx2action(self.last_action_idx)
        for i in range(n_agents):
            unit = self.get_unit_by_id(i)
            front_b, following_b = self.get_buffers_by_id(i)
            unit_id = unit.id
            health, health_time = unit.health
            under_m = unit.under_m
            expected_output = self.actions_output[actions[i]]
            last_output = unit.deque
            if expected_output == 0:
                saturation = 1
            else:
                saturation = last_output / expected_output

            feats[i, 0] = front_b  # front_b
            feats[i, 1] = following_b  # following_b
            feats[i, 2] = unit_id  # cell id
            feats[i, 3] = saturation #saturation

            ind = 4
            feats[i, ind] = health  # health
            ind += 1
            feats[i, ind] = health_time # health_time
            ind += 1
            feats[i, ind] = under_m #under m
            ind += 1
            feats[i, ind] = actions[i] # last action
            ind += 1
            feats[i, ind] = self.get_cost_by_id(i)

        agent_obs = feats.flatten()

        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs,
                                  self._episode_steps / self.episode_limit)
        if self.obs_profit:
            agent_obs = np.append(agent_obs, self.profit)

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))
        return agent_obs

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        return self.get_obs_agent()

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            return self.get_obs()
        else:
            return NotImplementedError

    def get_obs_size(self):
        """Returns the size of the observation."""
        nf_own = 9 #front buffer, following buffer, cell id, saturation, health state, health_time, under_m
        map_params = get_map_params(self.map_name)
        n_agents = map_params['n_agents']
        nf = nf_own * n_agents
        if self.obs_profit:
            nf += 1
        if self.obs_timestep_number:
            nf += 1
        return nf

    def get_state_size(self):
        """Returns the size of the state."""
        if self.obs_instead_of_state:
            return self.get_obs_size()
        else:
            return NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """
        Returns the available actions of an agents in a list
        """
        return self.sim.get_avail_agent_actions(agent_id)

    def get_avail_actions(self):
        """
        Returns the available actions of all agents in a list.
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        avail_actions = list(product(*avail_actions))
        output = [0] * self.n_actions
        for avail_action in avail_actions:
            idx = self.action2idx(avail_action)
            output[idx] = 1
        return output

    def get_unit_by_id(self, agent_id):
        return self.sim.machines[agent_id]

    def get_cost_by_id(self, agent_id):
        return self.sim.get_cost_agent(agent_id)

    # This function is wrong
    def get_buffers_by_id(self, agent_id):
        return self.sim.get_buffers_agent(agent_id)

    @property
    def products(self):
        return self.sim.products

    @property
    def profit(self):
        return self.sim.profit

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        actions_idx = actions
        self.last_action_idx = actions_idx
        self.sim.step(list(self.idx2action[actions_idx]))
        reward = self.reward_of_actions()
        self.accum_profit += self.profit
        self.current_profit += self.profit
        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, '-'))
        self._episode_steps += 1
        self._total_steps += 1

        terminated = False
        if self._episode_steps > self.episode_limit:
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self._episode_count += 1
            self.timeouts += 1

        info = {'profit':self.profit}
        return reward, terminated, info


    def reward_of_actions(self):
        reward = self.profit
        if self.reward_scale:
            return reward / self.reward_scale_rate
        return reward


    def get_avail_agent_actions(self, agent_id):
        avail_actions = []
        avail_logits = self.sim.get_avail_agent_actions(agent_id)
        for idx, logit in enumerate(avail_logits):
            if logit == 1:
                avail_actions.append(idx)
        return avail_actions



    def get_total_actions(self):
        return self.n_actions

    def get_stats(self):
        stats = {
            # "avg_profit": self.accum_profit / self._episode_count
            "cur_profit": self.current_profit
        }
        return stats
