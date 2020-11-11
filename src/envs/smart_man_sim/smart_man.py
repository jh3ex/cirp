from .core import Simulation
from .scenes import get_map_params

from envs.multiagentenv import MultiAgentEnv
from absl import logging
import numpy as np


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
        self.n_agents = map_params["n_agents"]
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
        self.n_actions = 3

        self.agents = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.random_init = None

    def _launch(self):
        """Launch the simulation"""
        self.sim = Simulation(self.map_name)

    def _reset(self):
        if self._episode_count < 909:
            random_init = None
        elif self._episode_count < 909 * 2:
            random_init = list(np.random.choice(6, 1, replace=False))
        elif self._episode_count < 909 * 3:
            random_init = list(np.random.choice(6, 2, replace=False))
        elif self._episode_count < 909 * 4:
            random_init = list(np.random.choice(6, 3, replace=False))
        elif self._episode_count < 909 * 5:
            random_init = list(np.random.choice(6, 4, replace=False))
        elif self._episode_count < 909 * 6:
            random_init = list(np.random.choice(6, 5, replace=False))
        else:
            random_init = list(np.arange(6, dtype=np.int32))
        self.random_init = random_init
        self.sim.reset(random_init)



    def reset(self):
        self._episode_steps = 0
        if self._episode_count == 0:
            self._launch()
        self._reset()
        self.current_profit = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        initial_actions = self.initial_step()
        self.step(initial_actions)

        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))
        return self.get_obs(), self.get_state()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        unit = self.get_unit_by_id(agent_id)

        nf_own = 7 #front buffer percentage, following buffer percentage, agent id, saturation, health state, state time, under_m
        if self.obs_last_action:
            nf_own += self.n_actions #action id
            if self.obs_own_cost:
                nf_own += 1 #cost of last action

        nf_al = 4 #front buffer, following buffer, agent id, and saturation

        if self.obs_all_health:
            # health state, state time, under_m
            nf_al += 3

        if self.obs_last_action:
            nf_al += self.n_actions
            if self.obs_all_cost:
                nf_al += 1

        ally_feats = np.zeros((self.n_agents - 1, nf_al), dtype=np.float32)
        own_feats = np.zeros(nf_own, dtype=np.float32)

        # Ally features
        al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
        for i, al_id in enumerate(al_ids):
            al_unit = self.get_unit_by_id(al_id)
            front_b, following_b = self.get_buffers_by_id(al_id)
            unit_id = al_unit.id
            health, health_time = al_unit.health
            under_m = al_unit.under_m
            expected_output = np.sum(np.array(self.actions_output) * self.last_action[i])
            last_output = al_unit.deque
            if expected_output == 0:
                saturation = 1
            else:
                saturation = last_output / expected_output

            ally_feats[i, 0] = front_b  # front_b
            ally_feats[i, 1] = following_b  # following_b
            ally_feats[i, 2] = unit_id  # cell id
            ally_feats[i, 3] = saturation

            ind = 4
            if self.obs_all_health:
                ally_feats[i, ind] = health  # health
                ind += 1
                ally_feats[i, ind] = health_time # health_time
                ind += 1
                ally_feats[i, ind] = under_m
                ind += 1
            if self.obs_last_action:
                ally_feats[i, ind:ind+self.n_actions] = self.last_action[i]
                ind += self.n_actions
                if self.obs_all_cost:
                    ally_feats[i, ind] = self.get_cost_by_id(i)

        # Own features
        front_b, following_b = self.get_buffers_by_id(agent_id)
        unit_id = unit.id
        health, health_time = unit.health
        under_m = unit.under_m
        last_output = unit.deque
        expected_output = np.sum(np.array(self.actions_output) * self.last_action[agent_id])
        if expected_output == 0:
            saturation = 1
        else:
            saturation = last_output / expected_output

        own_feats[0] = front_b
        own_feats[1] = following_b
        own_feats[2] = unit_id
        own_feats[3] = saturation
        ind = 4
        own_feats[ind] = health
        ind += 1
        own_feats[ind] = health_time
        ind += 1
        own_feats[ind] = under_m
        ind += 1
        if self.obs_last_action:
            own_feats[ind:ind+self.n_actions] = self.last_action[agent_id]
            ind += self.n_actions
            if self.obs_own_cost:
                own_feats[ind] = self.get_cost_by_id(agent_id)

        agent_obs = np.concatenate(
            (
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

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
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

        nf = 7 #front buffer, following buffer, id, health state, health_time, under_m, saturate
        if self.state_last_action:
            nf += 3 # action,
            if self.state_own_cost:
                nf += 1
        states = np.zeros((self.n_agents, nf))
        for i in range(self.n_agents):
            unit = self.get_unit_by_id(i)
            front_b, following_b = self.get_buffers_by_id(i)
            unit_id = unit.id
            health, health_time = unit.health
            under_m = unit.under_m
            last_output = unit.deque
            expected_output = np.sum(np.array(self.actions_output) * self.last_action[i])
            if expected_output == 0:
                saturation = 1
            else:
                saturation = last_output / expected_output

            states[i, 0] = front_b
            states[i, 1] = following_b
            states[i, 2] = unit_id
            states[i, 3] = health
            states[i, 4] = health_time
            states[i, 5] = under_m
            states[i, 6] = saturation
            ind = 7
            if self.state_last_action:
                states[i, ind:ind+self.n_actions] = self.last_action[i]
                ind += self.n_actions
                if self.state_own_cost:
                    states[i, ind] = self.get_cost_by_id(i)

        states = states.flatten()
        if self.state_timestep_number:
            states = np.append(agent_obs,
                                  self._episode_steps / self.episode_limit)

        if self.state_profit:
            states = np.append(states, self.profit)
        return states

    def get_obs_size(self):
        """Returns the size of the observation."""
        nf_own = 7 #front buffer, following buffer, cell id, saturation, health state, health_time, under_m
        if self.obs_last_action:
            nf_own += self.n_actions #action id
            if self.obs_own_cost:
                nf_own += 1 #cost of last action

        nf_al = 4 #front buffer, following buffer, cell id, and last output
        if self.obs_all_health:
            nf_al += 3
        if self.obs_last_action:
            nf_al += self.n_actions
            if self.obs_all_cost:
                nf_al += 1

        nf = nf_own + (self.n_agents - 1) * nf_al
        if self.obs_profit:
            nf += 1
        if self.obs_timestep_number:
            nf += 1
        return nf

    def get_state_size(self):
        """Returns the size of the state."""
        if self.obs_instead_of_state:
            return self.get_obs_size * self.n_agents

        nf_own = 7 #front buffer, following buffer, id, saturation, health state, health_time, under_m
        if self.state_last_action:
            nf_own += self.n_actions
            if self.state_own_cost:
                nf_own += 1
        nf = nf_own * self.n_agents
        if self.state_profit:
            nf += 1
        if self.state_timestep_number:
            nf += 1
        return nf

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
        return avail_actions

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
        actions_int = [int(a) for a in actions]
        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]
        self.sim.step(actions_int)
        reward = self.reward_of_actions()
        self.accum_profit += self.profit
        self.current_profit += self.profit
        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, '-'))
        self._episode_steps += 1
        self._total_steps += 1

        info = {}
        if self.random_init:
            info = {'random_init': len(self.random_init)}
        else:
            info = {'random_init': 0}
        terminated = False
        if self._episode_steps > self.episode_limit:
            terminated = True
            info['profit'] = self.current_profit
            if self.continuing_episode:
                info["episode_limit"] = True
            self._episode_count += 1
            self.timeouts += 1

        return reward, terminated, info


    def reward_of_actions(self):
        reward = self.profit
        if self.reward_scale:
            return reward / self.reward_scale_rate
        return reward


    def get_avail_agent_actions(self, agent_id):
        return self.sim.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        return self.n_actions

    def get_stats(self):
        stats = {
            # "avg_profit": self.accum_profit / self._episode_count
            "cur_profit": self.current_profit
        }
        return stats

    def initial_step(self):
        actions = []
        for i in range(self.n_agents):
            action_p = np.array([1., 0., 0.])
            valid_actions = self.get_avail_agent_actions(i)
            # print('valid_action is {}'.format(valid_actions))
            action_p = np.array(valid_actions, dtype=np.float32) * (action_p + 1e-9)
            p = action_p / np.sum(action_p)
            # print('p is {}'.format(p))
            action = np.random.choice(3, 1, p=p)[0]
            actions.append(action)
        return actions

    def close(self):
        return
