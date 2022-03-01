# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 07:04:31 2020
In this environment, machine tool follows hidden markov model.
@author: jingh
"""

# from envs.production_system.Buffer import Buffer, GrindingCB, IncomingBuffer
# from envs.production_system.Machine import GrindingRBE
# from envs.production_system.Product import Product


from Buffer import Buffer, GrindingCB, IncomingBuffer
from Machine import GrindingRBE
from Product import Product

import numpy as np
from copy import deepcopy


class production_rbe():
    def __init__(self, **env_args):
        """
        Env parameters
        """
        self.args = env_args

        # self.adj = self.args["adj"]
        self.stepsize = self.args["stepsize"]
        self.episode_limit = self.args["episode_limit"]
        self.sim_duration = self.args["sim_duration"]

        # Reward setting
        self.yield_reward = self.args["yield_reward"]
        self.defect_reward = self.args["defect_reward"]

        self.template_product = Product(n_feature=self.args["n_feature"],
                                        n_process=self.args["n_stage"],
                                        index=0)

        self._build_buffers()
        self._build_machines()
        self._build_actions()

        self.n_agents = len(self.machines)

        # self.seed = self.args["seed"]
        # self.RD = np.random.RandomState(seed=self.seed)


    def _build_actions(self):
        self.n_actions = 1
        v_ops = self.args["actions"]["v"]
        w_ops = self.args["actions"]["w"]
        a_ops = self.args["actions"]["a"]
        self.n_param_comb = len(v_ops) * len(w_ops) * len(a_ops)

        self.n_actions += self.n_param_comb

        # Add one more action if we need to make dress decision
        if not self.args["fixed_dress_schedule"]:
            self.n_actions += 2

        self.action_to_param = [[0, 0, 0]]

        for v in v_ops:
            for w in w_ops:
                for a in a_ops:
                    self.action_to_param.append([v, w, a])

        if not self.args["fixed_dress_schedule"]:
            self.action_to_param.append([0, 0, 0])
            self.action_to_param.append([0, 0, 0])


    def _build_buffers(self):
        # Build buffers
        self.buffers = {}
        self.buffers["incoming_buffer"] = IncomingBuffer(self.template_product, 1/self.args["obs_scale"].get("b_up", 1.0))
        self.buffers["completed_buffer"] = GrindingCB(self.args["q_star"], 1/self.args["obs_scale"].get("b_down", 1.0))

        for b in self.args["buffers"]:
            self.buffers[b] = Buffer(self.args["buffers"][b])

        return


    def _build_machines(self):
        # Build machines
        self.machines = []

        for idx, name in enumerate(self.args["machines"]):
            m = self.args["machines"][name]

            self.machines.append(GrindingRBE(tp=m["tp"],
                                            ep=m["ep"],
                                            p6=m["p6"],
                                            time_to_dress=m["time_to_dress"],
                                            fixed_dress_schedule=self.args["fixed_dress_schedule"],
                                            pass_to_dress=self.args["pass_to_dress"],
                                            p1=m["p1"],
                                            p2=m["p2"],
                                            p3=m["p3"],
                                            p4=m["p4"],
                                            p5=m["p5"],
                                            p5_scale=m["p5_scale"],
                                            features=None,
                                            stage=m["stage"],
                                            buffer_up=[self.buffers[x] for x in m["buffer_up"]],
                                            buffer_down=[self.buffers[x] for x in m["buffer_down"]],
                                            n_product_feature=self.args["n_feature"],
                                            name=name))



        return

    def _action_check(self, actions):
        avail_actions = self.get_avail_actions()

        for idx, a in enumerate(actions):
            assert avail_actions[idx][a] == 1, "At time [{}], agent {} is given an infeasible action {}".format(self.time, idx, a)


    def _last_action(self, actions):
        if self.args["obs_last_action"]:
            if self.args["last_action_one_hot"]:
                self.last_action = [[0] * self.n_actions] * self.n_agents
                for idx, a in enumerate(actions):
                    self.last_action[idx][a] = 1
            else:
                self.last_action = []
                for a in actions:
                    self.last_action.append(self.action_to_param[a])


    def step(self, actions):
        """ Returns reward, terminated, info """
        # raise NotImplementedError
        # Get the output and yield before this step

        # self._action_check(actions)

        self.steps += 1

        output_before, yield_before = self.buffers["completed_buffer"].output_and_yield()

        self._last_action(actions)

        decision_time = False

        while not decision_time:
            self.time += self.stepsize
            for idx, m in enumerate(self.machines):
                # Iterate over all machines
                # Quote machine current product and status
                status, product = m.quote(self.stepsize)
                # print(idx, m.status)
                # if m.w == 1:
                #     print("machine {} is down at time {}".format(idx, self.time))

                # self.w[idx].append(deepcopy(m.w))

                if status == "processing":
                    m.processing(self.stepsize)

                elif status == "to release":
                    for b in m.buffer_down:
                        if b.put(product, self.time):
                            m.release()
                            if not self.args["fixed_dress_schedule"]:
                                decision_time = True
                            # break
                elif (not self.args["fixed_dress_schedule"]) and status == "to dress":
                    if actions[idx] == self.n_actions - 1:
                        m.dress()
                    else:
                        m.status = "to load"

                elif status == "to load":
                    for b in m.buffer_up:
                        product = b.take()
                        if product is not None:
                            m.load(product)
                            # parameter_request[idx] = existing_feature
                            decision_time = True
                            # break

                elif status == "awaiting parameter":
                    parameters = self.action_to_param[actions[idx]]
                    m.set_process_parameter(parameters)

        self.output, self.yields = self.buffers["completed_buffer"].output_and_yield()
        output_step = self.output - output_before
        yield_step = self.yields - yield_before
        defect_step = output_step - yield_step

        reward = self.yield_reward * yield_step + self.defect_reward * defect_step
        if self.args["reward_scale"]:
            reward *= self.args["reward_scale_rate"]
        # self.episode_return += reward
        terminated = (self.steps >= self.episode_limit) or (self.time > self.sim_duration)
        # terminated = (self.time >= self.args["sim_duration"])
        info = {}
        if terminated:
            info["output"] = self.output
            info["yield"] = self.yields
            info["duration"] = self.time
            info["yield_rate"] = self.yields/self.time
            # info["reward"] = self.episode_return



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
        obs = []
        node_feature, need_decision, need_dress = self.machines[agent_id].get_node_feature()

        stage_one_hot = [0] * self.args["n_stage"]
        stage_one_hot[node_feature["stage"]] = 1
        node_feature["stage"] = stage_one_hot

        if not self.use_tool_belief:
            # Directly using observation instead of belief
            tool_ob_one_hot = [0] * self.args["n_tool_state"]
            tool_ob_one_hot[node_feature["tool_ob"]] = 1
            node_feature["tool_ob"] = tool_ob_one_hot

        for key, value in node_feature.items():
            scale = self.args["obs_scale"].get(key, 1)
            if isinstance(value, list):
                obs += [v * scale for v in value]
            else:
                obs.append(value * scale)

        if need_decision:
            obs.append(1)
        else:
            obs.append(0)

        if need_dress:
            obs.append(1)
        else:
            obs.append(0)

        if self.args["obs_last_action"]:
            if self.args["last_action_one_hot"]:
                obs += self.last_action[agent_id]
            else:
                scale = self.args["obs_scale"].get("actions", [1, 1, 1])
                for idx, action in enumerate(self.last_action[agent_id]):
                    obs.append(action * scale[idx])

        if self.args["obs_agent_id"]:
            obs.append(agent_id / self.n_agents)

        return np.array(obs, dtype=np.float32)


    def get_obs_size(self):
        """ Returns the shape of the observation """
        # raise NotImplementedError
        # Features from machine
        size = self.machines[0].get_feature_size() + self.args["n_stage"] + self.args["n_tool_state"] - 2

        # Stage is one-hot after processing

        # Features
        size += self.template_product.n_feature
        size += 2  # Include decision or not

        if self.args["obs_last_action"]:
            if self.args["last_action_one_hot"]:
                size += self.n_actions
            else:
                size += len(self.action_to_param[0])

        if self.args["obs_agent_id"]:
            size += 1

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
        obs, need_decision, need_dress = self.machines[agent_id].get_node_feature()

        # assert not need_decision, "At time {}, agent {} needs decision".format(self.time, agent_id)

        if self.args["fixed_dress_schedule"]:
            if need_decision:
                avail_agent_actions = [0] + [1] * self.n_param_comb
            else:
                avail_agent_actions = [1] + [0] * self.n_param_comb
        else:
            if need_decision:
                avail_agent_actions = [0] + [1] * self.n_param_comb + [0] + [0]
            elif need_dress:
                avail_agent_actions = [0] + [0] * self.n_param_comb + [1] + [1]
            else:
                avail_agent_actions = [1] + [0] * self.n_param_comb + [0] + [0]


        return avail_agent_actions


    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def reset(self, seed=None):
        """ Returns initial observations and states"""
        # raise NotImplementedError
        # Random seed for simulation

        if seed:
            np.random.seed(seed)
        # Simulation time horizon
        self.stepsize = self.args["stepsize"]
        self.steps = 0
        self.time = 0.0
        # self.episode_return = 0.0

        self.output, self.yields = 0, 0

        if self.args["obs_last_action"]:
            if self.args["last_action_one_hot"]:
                self.last_action = [[0] * self.n_actions] * self.n_agents
            else:
                self.last_action = [self.action_to_param[0]] * self.n_agents

        # Initialize everthing
        for m in self.machines:
            m.initialize()

        for b in self.buffers:
            self.buffers[b].initialize()



        self.w = [[0]] * self.n_agents

    def get_stats(self):
        # Add per warning during running
        self.stats = {"output": self.output,
                      "yield": self.yields,
                      "duration": self.time,
                      "yield_rate": self.yields/self.time,
                      "compliant_ratio": self.yields/self.output}

        return self.stats



    def render(self):
        raise NotImplementedError

    def close(self):
        # raise NotImplementedError
        pass

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


    def param_lookup(self, n):
        return self.action_to_param[n]



def fixed_controller(action, default):
    fixed_action = deepcopy(action)
    for i in range(len(fixed_action)):
        if fixed_action[i] != 0:
            fixed_action[i] = default

    return fixed_action


def fixed_controller_agent(action, default):
    fixed_action = deepcopy(action)
    for i in range(len(fixed_action)):
        if fixed_action[i] != 0:
            fixed_action[i] = default[i]

    return fixed_action





if __name__ == "__main__":
    from types import SimpleNamespace as SN
    import yaml
    import numpy as np
    import matplotlib.pyplot as plt


    with open('../../config/envs/production_rbe.yaml', 'r') as f:
        _config = yaml.load(f, Loader=yaml.FullLoader)
    env_config = _config['env_args']
    env = production_rbe(**env_config)




    y = []

    o = []

    d = []

    ratio = []

    for i in range(1):

        env.reset()
        terminated = False



        while not terminated:
            # print(env.machines[0].status)
            rand_action = np.random.rand(env.n_agents, env.n_actions)
            # print('At time [{}], available actions are {}'.format(env.time, env.get_avail_actions()))
            # print("state is {}".format(env.get_state()))
            logits = rand_action * np.array(env.get_avail_actions())
            p = logits / np.sum(logits, axis=1, keepdims=True)
            rand_action = np.argmax(p, axis=1)

            # rand_action = fixed_controller(rand_action, 1)
            # rand_action = fixed_controller_agent(rand_action, [5, 1, 21, 1, 1])

            r, terminated, info = env.step(rand_action)
            # print(r)
        print(info)
        y.append(info["yield"])
        o.append(info["output"])
        d.append(o[-1] - y[-1])
        ratio.append(y[-1] / o[-1])
