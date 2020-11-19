# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 07:04:31 2020
In this environment, actions are discrete.
@author: jingh
"""
from envs.multiagentenv import MultiAgentEnv
from Buffer import Buffer
from GrindingRF import GrindingRF
from GrindingCB import GrindingCB
from Product import Product
from IncomingBuffer import IncomingBuffer


class production_discrete(MultiAgentEnv):
#class production_discrete():
    def __init__(self, **env_args):
        """
        Env parameters
        """
        self.args = env_args

        # self.adj = self.args["adj"]
        self.stepsize = self.args["stepsize"]
        self.episode_limit = self.args["episode_limit"]

        # Reward setting
        self.yield_reward = self.args["yield_reward"]
        self.defect_reward = self.args["defect_reward"]

        self.template_product = Product(n_feature=self.args["n_feature"],
                                        n_process=self.args["n_process"],
                                        index=0)

        self._build_buffers()
        self._build_machines()
        self._build_actions()

        self.n_agents = len(self.machines)

        self.seed = self.args["seed"]
        self.RD = np.random.RandomState(seed=self.seed)


    def _build_actions(self):
        v_ops = self.args["actions"]["v"]
        w_ops = self.args["actions"]["w"]
        a_ops = self.args["actions"]["a"]

        self.n_actions = len(v_ops) * len(w_ops) * len(a_ops) + 1

        self.action_to_param = [[0, 0, 0]]

        for v in v_ops:
            for w in w_ops:
                for a in a_ops:
                    self.action_to_param.append([v, w, a])
        return

    def _build_buffers(self):
        # Build buffers
        self.buffers = {}
        self.buffers["incoming_buffer"] = IncomingBuffer(self.template_product)
        self.buffers["completed_buffer"] = GrindingCB(q_star=self.args["q_star"])

        for b in self.args["buffers"]:
            self.buffers[b] = Buffer(self.args["buffers"][b])

        return


    def _build_machines(self):
        # Build machines
        self.machines = []

        for idx, name in enumerate(self.args["machines"]):
            m = self.args["machines"][name]
            self.machines.append(GrindingRF(p1=m["p1"],
                                            p2=m["p2"],
                                            p3=m["p3"],
                                            p4=m["p4"],
                                            p5=m["p5"],
                                            features=None,
                                            stage=m["stage"],
                                            buffer_up=[self.buffers[x] for x in m["buffer_up"]],
                                            buffer_down=[self.buffers[x] for x in m["buffer_down"]],
                                            MTTR_step=m["MTTR_step"],
                                            MTBF_step=m["MTBF_step"],
                                            n_product_feature=self.args["n_feature"],
                                            name=name))



        return

    def step(self, actions):
        """ Returns reward, terminated, info """
        # raise NotImplementedError
        # Get the output and yield before this step
        self.steps += 1

        output_before, yield_before = self.buffers["completed_buffer"].output_and_yield()

        self.last_action = [[0] * self.n_actions] * self.n_agents
        for idx, a in enumerate(actions):
            self.last_action[idx][a] = 1

        # parameter_request = [None] * self.n_agents

        decision_time = False

        while not decision_time:
            self.time += self.stepsize
            for idx, m in enumerate(self.machines):
                # Iterate over all machines
                # Quote machine current product and status
                status, product = m.quote()
                if status == "processing":
                    m.processing(self.stepsize)
                elif status == "to release":
                    for b in m.buffer_down:
                        if b.put(product, self.time):
                            m.release()
                            break
                elif status == "to load":
                    for b in m.buffer_up:
                        product = b.take()
                        if product is not None:
                            existing_feature = m.load(product)
                            # parameter_request[idx] = existing_feature
                            decision_time = True
                            break
                elif status == "awaiting parameter":
                    # parameters = self.action_to_param[actions[idx].index(1)]
                    parameters = self.action_to_param[actions[idx]]
                    m.set_process_parameter(parameters)

        output_after, yield_after = self.buffers["completed_buffer"].output_and_yield()
        output_step = output_after - output_before
        yield_step = yield_after - yield_before
        defect_step = output_step - yield_step

        reward = self.yield_reward * yield_step + self.defect_reward * defect_step
        self.episode_return += reward
        terminated = (self.steps >= self.episode_limit)
        info = {}
        if terminated:
            info["output"] = output_after
            info["yield"] = yield_after
            info["reward"] = self.episode_return


        return reward, terminated, info


    def get_obs(self):
        """ Returns all agent observations in a list """
        # raise NotImplementedError
        obs_all = []
        for agent_id in range(self.n_agents):
            obs_all.append(self.get_obs_agent(agent_id))
        return obs_all

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        # raise NotImplementedError
        obs, need_decision = self.machines[agent_id].get_node_feature()
        if need_decision:
            obs.append(1.0)
        else:
            obs.append(0.0)

        if self.args["obs_last_action"]:
            obs += self.last_action[agent_id]

        return obs


    def get_obs_size(self):
        """ Returns the shape of the observation """
        # raise NotImplementedError
        # Features from machine
        size = self.machines[0].get_feature_size()
        # Features
        size += self.template_product.n_feature
        size += 1  # Include agent id

        if self.args["obs_last_action"]:
            size += self.n_actions

        return size


    def get_state(self):
        # raise NotImplementedError

        if self.args["obs_instead_of_state"]:
            obs_all = self.get_obs()
            state = [item for sublist in obs_all for item in sublist]
            return state
        else:
            return NotImplementedError


    def get_state_size(self):
        """ Returns the shape of the state"""
        # raise NotImplementedError

        if self.args["obs_instead_of_state"]:
            return len(self.machines) * self.get_obs_size()
        else:
            return NotImplementedError

    def get_avail_actions(self):
        # raise NotImplementedError
        avail_actions = [self.get_avail_agent_actions(agent_id) for agent_id in range(self.n_agents)]
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        # raise NotImplementedError
        obs, need_decision = self.machines[agent_id].get_node_feature()
        if need_decision:
            return [0] + [1] * (self.n_actions - 1)
        else:
            return [1] + [0] * (self.n_actions - 1)


    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        # raise NotImplementedError
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        # raise NotImplementedError
        # Random seed for simulation


        # Simulation time horizon
        self.stepsize = self.args["stepsize"]
        self.steps = 0
        self.time = 0.0
        self.episode_return = 0.0

        self.last_action = [[0] * self.n_actions] * self.n_agents

        # Initialize everthing
        for m in self.machines:
            m.initialize(self.RD)

        for b in self.buffers:
            self.buffers[b].initialize()

        # self.incoming_buffer.initialize()
        # self.completed_buffer.initialize()



    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info


if __name__ == "__main__":
    from types import SimpleNamespace as SN
    import yaml
    import numpy as np

    with open('D:/OneDrive/Script/graph/pymarl_adaptive_graph/src/config/envs/production.yaml', 'r') as f:
        _config = yaml.load(f)
    env_config = _config['env_args']
    env = production_discrete(**env_config)
    env.reset()
    terminated = False

    while not terminated:
        rand_action = np.random.rand(env.n_agents, env.n_actions)
        print('At time [{}], available actions are {}'.format(env.time, env.get_avail_actions()))
        print("state is {}".format(env.get_state()))
        logits = rand_action * np.array(env.get_avail_actions())
        p = logits / np.sum(logits, axis=1, keepdims=True)
        rand_action = np.argmax(p, axis=1)
        r, terminated, info = env.step(rand_action)
        print(r)
    print(info)
    print(env.get_env_info())
