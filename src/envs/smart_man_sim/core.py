import time
import numpy as np
import random
import math
from absl import logging
from .scenes import get_map_params
from functools import partial
import math



class Transition():
    """
    machine health states transition rules:
    -There are 4 health states: pre-mature, mature, slightly damaged, severely damaged
    """
    def __init__(self, tran_matrix, length=10, exp=False, schedule=False):
        self._init_matrix(tran_matrix)
        self.schedule = schedule
        self.length = length
        self.exp = exp
        if schedule:
            if not self.exp:
                self.init = np.ones_like(self.T)
                self.init[self.T==0] = 0
                self.decay = (self.init - self.T) / length
            else:
                return NotImplementedError
                # self.init = np.ones_like(self.T)
                # print(self.T.shape, self.init.shape)
                # self.init[self.T==0] = 0
                # self.T[self.T == 0] = 1
                # self.exp_scaling = (-1) * self.length / np.log(self.T)

    def _init_matrix(self, tran_matrix):
        """
        T = [[[], [], [], []],
             [[], [], [], []],
             [[], [], [], []]]
        """
        self.T = tran_matrix

    def transit(self, init_state, action, steps=None):
        if not self.schedule:
            T_a = self.T[action]
            p = T_a[init_state]
            next_state = np.random.choice(4, 1, p=p)[0]
            return next_state
        else:
            if not self.exp:
                # if steps[init_state] > self.length:
                #     steps = self.length
                T = self.init - self.decay * np.array(steps)
                T_a = T[action]
                p = T_a[init_state]
                p /= np.sum(p)
                next_state = np.random.choice(4, 1, p=p)[0]
                # print('The state is {} and the probability of not trans is {}, we trans {}, the action is {}'.format(init_state, p, trans, action))
                return next_state
            else:
                return NotImplementedError
                # T = np.exp(- np.array(steps) / self.exp_scaling)
                # T_a = T[action]
                # p = T_a[init_state]
                # if p == 1.:
                #     p = 0
                # else:
                #     p = max(self.T[action, init_state], p)
                # trans = np.random.choice(2, 1, p=[p, 1-p])
                # # print('The state is {} and the probability of not trans is {}, we trans {}, the action is {}'.format(init_state, p, trans, action))
                # if trans:
                #     next_state = init_state + 1
                #     if next_state > 3:
                #         next_state %= 3
                # else:
                #     next_state = init_state
                # return next_state

class Continue_Transition():
    def __init__(self, dist, first_params, second_params, lower_bounds):
        assert type(dist) == str, "dist must be string"
        self.dist_name = dist
        self.first_params = first_params
        self.second_params = second_params
        self.lower_bounds = lower_bounds
        if dist == 'log-normal':
            self.dist = np.random.lognormal
        elif dist == 'exponential':
            self.dist = np.random.exponential
        elif dist == 'gamma':
            self.dist = np.random.gamma
        elif dist == 'static':
            self.dist = None
        else:
            raise ValueError("{} is not a predefined distributions, which has to be in [log-normal, exponential, gamma]".format(dist))

    def init_trans(self, init_state):
        first_param = self.first_params[init_state]
        second_param = self.second_params[init_state]
        lower_bound = self.lower_bounds[init_state]
        if self.dist_name == 'log-normal':
            mean = first_param
            sigma = second_param
            self.end_time = max(lower_bound, math.ceil(self.dist(mean, sigma)))
        elif self.dist_name == 'exponential':
            offset = first_param
            scale = second_param
            self.end_time = max(lower_bound, math.ceil(offset + self.dist(scale)))
        elif self.dist_name == 'gamma':
            shape = first_param
            scale = second_param
            self.end_time = max(lower_bound, math.ceil(self.dist(shape, scale)))
        elif self.dist_name == 'static':
            shape = first_param
            scale = second_param
            self.end_time = first_param
        else:
            raise ValueError("{} is not a predefined distributions, which has to be in [log-normal, exponential, gamma]".format(self.dist_name))
        steps = random.randint(0, self.end_time-1)
        self.end_time -= steps
        return steps

    def transit(self, init_state, action, steps=None):
        if steps[init_state] == 0:
            raise ValueError("wrong steps!")
        if init_state != 3:
            if steps[init_state] == 1:
                first_param = self.first_params[init_state]
                second_param = self.second_params[init_state]
                lower_bound = self.lower_bounds[init_state]
                if self.dist_name == 'log-normal':
                    mean = first_param
                    sigma = second_param
                    self.end_time = max(lower_bound, math.ceil(self.dist(mean, sigma)))
                elif self.dist_name == 'exponential':
                    offset = first_param
                    scale = second_param
                    self.end_time = max(lower_bound, math.ceil(offset + self.dist(scale)))
                elif self.dist_name == 'gamma':
                    shape = first_param
                    scale = second_param
                    self.end_time = max(lower_bound, math.ceil(self.dist(shape, scale)))
                elif self.dist_name == 'static':
                    shape = first_param
                    self.end_time = first_param
                else:
                    raise ValueError("{} is not a predefined distributions, which has to be in [log-normal, exponential, gamma]".format(self.dist_name))

            if action != 1:
                self.end_time -= 1
            else:
                self.end_time -= .75
            if self.end_time <= 0:
                init_state += 1
                if init_state > 3:
                    init_state %= 3
        # else:
        #     if init_state != 3:
        #         self.end_time -= 1
        #         if self.end_time == 0:
        #             init_state += 1
        #             if init_state > 3:
        #                 init_state %= 3
        else:
            init_state = 0
        return init_state







ACTION = {0: 1000,
          1: 0,
          2: (12, 3)}

class Machine():
    def __init__(self, id, T, cell_id, config, time_base=False):
        self.T = T
        self.id = id
        self.cell_id = cell_id
        self.under_m = False
        self.deque = 0
        self.anti_jobs = 0
        self.action = None
        self.m_cd = 0
        self.ACTION = config['actions']
        self.COST = config['costs']
        self.restart = 0
        self.h_tracker = [[],[],[],[]]
        self.init_time_base = time_base
        if time_base:
            self.counter = self.init_time_base

    def reset(self, random_init):
        self.under_m = False
        self.deque = 0
        self.anti_jobs = 0
        self.action = None
        self.m_cd = 0
        self.h_tracker = [[],[],[],[]]
        self.random_init = random_init
        self.init_h_state(random_init)
        if self.init_time_base:
            self.counter = self.init_time_base




    def step(self, action):
        # print("machine {} takes action {} with {} time in health state {}".format(self.id, action, self.state_time[self.h_state], self.h_state))
        if action == 0:
            if self.init_time_base:
                self.counter -= 1
            assert self.h_state != 3
            self.action = action
            n = self.ACTION[self.action]
            self.request_parts(n)
            self.state_time[self.h_state] += 1
        elif action == 1:
            if self.init_time_base:
                self.counter -= 1
            assert self.h_state != 3
            self.action = action
            n = self.ACTION[self.action]
            self.request_parts(n)
            self.state_time[self.h_state] += 1
        elif action == 2:
            # assert self.h_state != 0 and self.h_state != 1
            self.action = action
            self.request_parts(0)
            self.register_m()
            self.state_time[self.h_state] += 1
        else:
            raise ValueError('action [%d] is out of range'%action)

    def init_h_state(self, random_init):
        if type(self.T) == Continue_Transition:
            if random_init:
                # print("Machine {} is random inited".format(self.id))
                self.h_state = random.randint(0, 3)
                steps = self.T.init_trans(self.h_state)
                self.state_time = [0] * 4
                self.state_time[self.h_state] = steps
                if self.h_state == 3:
                    self.register_m()
                    steps = random.randint(0, self.m_cd-1)
                    self.m_cd -= steps
                    assert self.m_cd > 0
                    self.state_time = [0] * 4
                    self.state_time[self.h_state] = steps
            else:
                self.h_state = 0
                self.state_time = [0] * 4
        else:
            raise ValueError("We currently only support continuous transitions")


    @property
    def health(self):
        return self.h_state, self.state_time[self.h_state]

    def request_parts(self, n):
        self.request_jobs = n

    def recieve_parts(self, n):
        self.anti_jobs = n

    def register_m(self):
        self.under_m = True
        self.anti_jobs = 0
        self.h_state = 3
        self.request_jobs = 0
        if self.m_cd == 0:
            self.m_cd = max(1, math.floor(np.random.normal(12, 3)))

    def proceed(self):
        if not self.under_m:
            self.deque = self.anti_jobs
            self.anti_jobs = 0
            self.request_jobs = 0
            new_h_state = self.T.transit(self.h_state, self.action, self.state_time)
            if new_h_state != self.h_state:
                self.h_tracker[self.h_state].append(self.state_time[self.h_state])
            self.h_state = new_h_state
        else:
            self.m_cd -= 1
            assert self.m_cd >= 0, 'self.m_cd value is {}'.format(self.m_cd)
            self.deque = self.anti_jobs
            if self.m_cd == 0:
                self.under_m = False
                new_h_state = self.T.transit(self.h_state, self.action, self.state_time)
                assert new_h_state != self.h_state, 'new state {} should be different from the original state {}'.format(new_h_state, self.h_state)
                if new_h_state != self.h_state:
                    self.h_tracker[self.h_state].append(self.state_time[self.h_state])
                self.h_state = new_h_state
                if self.init_time_base:
                    self.counter = self.init_time_base
                self.state_time = [0] * 4
                self.restart += 1

    @property
    def valid_actions(self):
        if not self.init_time_base:
            if self.h_state == 3 or self.under_m:
                valid_actions = [0., 0., 1.]
            elif self.h_state == 0 or self.h_state == 1:
                valid_actions = [1., 1., 1.]
            elif self.h_state == 2:
                valid_actions = [1., 1., 1.]
            else:
                raise ValueError("we are in wrong {} state".format(self.h_state))
        else:
            if self.counter == 0:
                valid_actions = [0., 0., 1.]
            else:
                if self.h_state == 3 or self.under_m:
                    valid_actions = [0., 0., 1.]
                elif self.h_state == 0 or self.h_state == 1:
                    valid_actions = [1., 1., 0.]
                elif self.h_state == 2:
                    valid_actions = [1., 1., 1.]
                else:
                    raise ValueError("we are in wrong {} state".format(self.h_state))
        return valid_actions

    @property
    def cost(self):
        if self.action == 0 and self.h_state != 3:
            cost = self.COST[0]
            # print('Agent {} took action {} and incur {} cost'.format(self.id, self.action, cost))
        elif self.action == 1 and self.h_state != 3:
            cost =  self.COST[1]
            # print('Agent {} took action {} and incur {} cost'.format(self.id, self.action, cost))
        elif self.h_state == 3 and self.state_time[3] == 0:
            cost = self.COST[-1]
            if self.action == 0:
                cost += self.COST[0]
                # print('Agent {} took action {}, and breakdown, and incur {} cost'.format(self.id, self.action, cost))
            elif self.action == 1:
                cost += self.COST[1]
                # print('Agent {} took action {}, and breakdown, and incur {} cost'.format(self.id, self.action, cost))
            else:
                raise ValueError('self.action cannot take {}, the current state time is {}, state is {}, m_cd is {}, under_m is {}'.format(self.action, self.state_time, self.h_state, self.m_cd, self.under_m))
        elif self.action == 2:
            cost = self.COST[2]
            # print('Agent {} took action {} and incur {} cost'.format(self.id, self.action, cost))
        else:
            raise ValueError("Our agent going to wrong state, action pair {}{}".format(self.h_state, self.action))
        return cost


# class Machine():
#     def __init__(self, id, T, cell_id, config, time_base=False):
#         self.T = T
#         self.id = id
#         self.cell_id = cell_id
#         self.under_m = False
#         self.deque = 0
#         self.anti_jobs = 0
#         self.action = None
#         self.m_cd = 0
#         self.ACTION = config['actions']
#         self.COST = config['costs']
#         self.h_history = np.zeros_like(self.COST)
#         self.restart = 0
#         # self.h_tracker = [[],[],[],[]]
#         self.init_h_state()
#         self.init_time_base = time_base
#         if time_base:
#             self.counter = self.init_time_base
#
#
#     def step(self, action):
#         if action == 0:
#             if self.init_time_base:
#                 self.counter -= 1
#             assert self.h_state != 3
#             self.action = action
#             n = self.ACTION[self.action]
#             self.request_parts(n)
#             self.state_time[self.h_state] += 1
#         elif action == 1:
#             if self.init_time_base:
#                 self.counter -= 1
#             assert self.h_state != 3
#             self.action = action
#             n = self.ACTION[self.action]
#             self.request_parts(n)
#             self.state_time[self.h_state] += 1
#         elif action == 2:
#             self.action = action
#             self.request_parts(0)
#             self.register_m()
#             self.state_time[self.h_state] += 1
#         else:
#             raise ValueError('action [%d] is out of range'%action)
#         self.h_history[self.h_state] += 1
#
#     def init_h_state(self):
#         if type(self.T) == Continue_Transition:
#             self.h_state = random.randint(0, 3)
#             steps = self.T.init_trans(self.h_state)
#             self.state_time = [0] * 4
#             self.state_time[self.h_state] = steps
#             if self.h_state == 3:
#                 self.register_m()
#                 steps = random.randint(0, self.m_cd-1)
#                 self.m_cd -= steps
#                 self.state_time = [0] * 4
#                 self.state_time[self.h_state] = steps
#         else:
#             self.h_state = 0
#             self.state_time = [0] * 4
#
#     @property
#     def health(self):
#         return self.h_state, self.state_time[self.h_state]
#
#     def request_parts(self, n):
#         self.request_jobs = n
#
#     def recieve_parts(self, n):
#         self.anti_jobs = n
#
#     def register_m(self):
#         self.under_m = True
#         self.anti_jobs = 0
#         self.h_state = 3
#         self.request_jobs = 0
#         if self.m_cd == 0:
#             self.m_cd = math.floor(np.random.normal(12, 3))
#
#     def proceed(self):
#         if not self.under_m:
#             self.deque = self.anti_jobs
#             self.anti_jobs = 0
#             self.request_jobs = 0
#             new_h_state = self.T.transit(self.h_state, self.action, self.state_time)
#             # if new_h_state != self.h_state:
#             #     self.h_tracker[self.h_state].append(self.state_time[self.h_state])
#             self.h_state = new_h_state
#         else:
#             self.m_cd -= 1
#             self.deque = self.anti_jobs
#             if self.m_cd == 0:
#                 self.under_m = False
#                 new_h_state = self.T.transit(self.h_state, self.action, self.state_time)
#                 # if new_h_state != self.h_state:
#                 #     self.h_tracker[self.h_state].append(self.state_time[self.h_state])
#                 self.h_state = new_h_state
#                 if self.init_time_base:
#                     self.counter = self.init_time_base
#                 self.state_time = [0] * 4
#                 self.restart += 1
#
#     @property
#     def valid_actions(self):
#         if self.h_state == 3 or self.under_m:
#             valid_actions = [0., 0., 1.]
#         else:
#             if self.init_time_base:
#                 if self.counter == 0:
#                     valid_actions = [0., 0., 1.]
#             valid_actions = [1] * 3
#         return valid_actions
#
#     @property
#     def cost(self):
#         if self.action == 0 and self.h_state != 3:
#             cost = self.COST[0]
#             # print('Agent {} took action {} and incur {} cost'.format(self.id, self.action, cost))
#         elif self.action == 1 and self.h_state != 3:
#             cost =  self.COST[1]
#             # print('Agent {} took action {} and incur {} cost'.format(self.id, self.action, cost))
#         elif self.h_state == 3 and self.state_time[3] == 0:
#             cost = self.COST[-1]
#             if self.action == 0:
#                 cost += self.COST[0]
#                 # print('Agent {} took action {}, and breakdown, and incur {} cost'.format(self.id, self.action, cost))
#             elif self.action == 1:
#                 cost += self.COST[1]
#                 # print('Agent {} took action {}, and breakdown, and incur {} cost'.format(self.id, self.action, cost))
#             else:
#                 raise ValueError('self.action cannot take {}'.format(self.action))
#         elif self.action == 2:
#             cost = self.COST[2]
#             # print('Agent {} took action {} and incur {} cost'.format(self.id, self.action, cost))
#         else:
#             raise ValueError("Our agent going to wrong state, action pair {}{}".format(self.h_state, self.action))
#         return cost


class Cell():
    def __init__(self, id):
        self.id = id
        self.deque = 0
        self.anti_jobs = 0
        self.p_cell = None
        self.f_cell = None
    def add_machines(self, m_list):
        self.machines = m_list

    def add_cells(self, p_cell=None, f_cell=None):
        self.p_cell = p_cell
        self.f_cell = f_cell
        if self.p_cell:
            self.p_cell.f_cell = self
        if self.f_cell:
            self.f_cell.p_cell = self

    def assign_jobs(self):
        self.deque = 0
        if not self.p_cell:
            assert self.anti_jobs >= 0, 'anti_jobs {} should be always greater than 0'.format(self.anti_jobs)
            recieve_requests = np.sum(list(map(lambda x: x.request_jobs, self.machines)))
            self.anti_jobs += recieve_requests
            for machine in self.machines:
                machine.recieve_parts(machine.request_jobs)
            assert self.anti_jobs >= np.sum(list(map(lambda x: x.anti_jobs, self.machines))), 'anti_jobs is {}, machines in cell {} actually get {}'.format(self.anti_jobs, self.id, np.sum(list(map(lambda x: x.anti_jobs, self.machines))))
            if self.f_cell:
                self.f_cell.anti_jobs += np.sum(list(map(lambda x: x.anti_jobs, self.machines)))
        else:
            if self.anti_jobs > 0:
                recieve_requests = np.sum(list(map(lambda x: x.request_jobs, self.machines)))
                if self.anti_jobs >= recieve_requests:
                    for machine in self.machines:
                        machine.recieve_parts(machine.request_jobs)
                    if self.f_cell:
                        self.f_cell.anti_jobs += np.sum(list(map(lambda x: x.anti_jobs, self.machines)))
                else:
                    request_jobs = np.array(list(map(lambda x: x.request_jobs, self.machines)), dtype=np.float32)
                    jobs_pool = np.zeros_like(self.machines, dtype=np.float32)
                    while np.sum(request_jobs - jobs_pool) > 0:
                        p = (request_jobs - jobs_pool) / np.sum(request_jobs - jobs_pool)
                        idx = np.random.choice(len(self.machines), 1, p=p)[0]
                        jobs_pool[idx] += 1.

                    for idx, machine in enumerate(self.machines):
                        machine.recieve_parts(jobs_pool[idx])
                    if self.f_cell:
                        self.f_cell.anti_jobs += np.sum(list(map(lambda x: x.anti_jobs, self.machines)))

    def proceed(self):
        for m in self.machines:
            m.proceed()
        self.deque = np.sum(list(map(lambda x: x.deque, self.machines)))
        self.anti_jobs -= self.deque
        # assert self.anti_jobs >= 0, 'anti_jobs is {}, and deques is {}'.format(self.anti_jobs, self.deque)

    @property
    def buffer_size(self):
        return self.anti_jobs

    def reset(self):
        self.deque = 0
        self.anti_jobs = 0





class Simulation():
    def __init__(self, map_name, time_base=False):
        self._initialize(map_name, time_base)

    def _initialize(self, map_name, time_base):
        config = get_map_params(map_name)
        cell_ids = config['cells']
        machine_ids = config['machines']
        self.sale_price = config['sale_price']
        continuous = config['continuous_trans']

        machine_ids = np.array(machine_ids).reshape([len(cell_ids), -1])
        self.machines = []
        self.cells = []
        for i in range(machine_ids.shape[0]):
            cell_id = i
            self.cells.append(Cell(cell_id))
            for j in range(machine_ids.shape[1]):
                machine_id = machine_ids[i, j]
                if not continuous:
                    transition = config['transitions'][i]
                    T = Transition(transition, schedule=False)
                else:
                    T = Continue_Transition(config['dist'], config['first_params'], config['second_params'], config['lower_bounds'])
                self.machines.append(Machine(machine_id, T, cell_id, config, time_base))
            self.cells[-1].add_machines(self.machines[-machine_ids.shape[1]:])
            if i > 0:
                p_cell = self.cells[i-1]
                self.cells[-1].add_cells(p_cell)

    def reset(self, random_init_sim):
        for cell in self.cells:
            cell.reset()
        random_list = [0] * len(self.machines)
        if random_init_sim:
            for ele in random_init_sim:
                random_list[ele] = 1
        for idx, machine in enumerate(self.machines):
            machine.reset(random_list[idx])


    def step(self, actions):
        for idx, machine in enumerate(self.machines):
            machine.step(actions[idx])
        for cell in self.cells:
            cell.assign_jobs()
        for cell in self.cells:
            cell.proceed()

    def get_avail_agent_actions(self, agent_id):
        return self.machines[agent_id].valid_actions

    @property
    def products(self):
        final_cell = self.cells[-1]
        products = final_cell.deque
        return products

    @property
    def profit(self):
        products = self.products
        cost = np.sum(list(map(lambda x: x.cost, self.machines)))
        return products * self.sale_price - cost

    def get_buffers_agent(self, agent_id):
        total_buffer = np.sum(list(map(lambda x:x.buffer_size, self.cells)))
        if total_buffer == 0:
            return 0., 0.
        agent = self.machines[agent_id]
        cell_id = agent.cell_id
        front_buffer = self.cells[cell_id].buffer_size
        following_buffer = 0
        if cell_id + 1 < len(self.cells) -1:
            following_buffer = self.cells[cell_id+1].buffer_size
        return front_buffer / total_buffer, following_buffer / total_buffer

    def get_cost_agent(self, agent_id):
        return self.machines[agent_id].cost



if __name__ == '__main__':
    from scenes import get_map_params
    import matplotlib.pyplot as plt
    import numpy as np
    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    def pie_plot(*args, **kw_args):
        labels = kw_args['labels']
        colors = tableau20[:len(labels)]
        sizes = args

        def func(pct, allvals):
            absolute = pct/100.*np.sum(allvals)
            return "{:.1f}%\n({:.1f} unit time)".format(pct, absolute)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, colors=colors, labels=labels, autopct=lambda pct:func(pct, sizes),
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.show()
    map_name = 263
    profit = 0
    sim = Simulation(map_name)
    for i in range(20):
        random_init_sim = list(np.random.choice(6, 3, replace=False))
        # print(random_init_sim)
        print("At iteration {}".format(i))
        sim.reset(random_init_sim)
        # sim.step([0]*6)
        num = 0

        while num < 64:
            actions = []
            for i in range(6):
                action_p = np.array([1., 0., 0.])
                valid_actions = sim.get_avail_agent_actions(i)
                # print('valid_action is {}'.format(valid_actions))
                action_p = np.array(valid_actions, dtype=np.float32) * (action_p + 1e-9)
                p = action_p / np.sum(action_p)
                # print('p is {}'.format(p))
                action = np.random.choice(3, 1, p=p)[0]
                actions.append(action)
            sim.step(actions)
            # print("Actions are {}".format(actions))
            # print("States are {}".format([machine.h_state for machine in sim.machines]))
            num += 1
            # print(sim.profit)
            profit += sim.profit
        # cells = []
        # for m in sim.machines:
        #     cells.append(m.h_history/m.restart)
        #     print("States history are {}".format(m.h_history/m.restart))
        #
    for i in range(4):
        h = np.concatenate(list(map(lambda x:np.array(x.h_tracker[i]), sim.machines[:3])))
        # if i == 1:
        #     print(h)
        print("Health state %d has mean %.3f, std %.3f"%(i, np.mean(h), np.std(h)))

    for i in range(4):
        h = np.concatenate(list(map(lambda x:np.array(x.h_tracker[i]), sim.machines[3:])))
        print("Health state %d has mean %.3f, std %.3f"%(i, np.mean(h), np.std(h)))

        # print(np.mean(cells[3:], axis=0))
        # #pie_plot(*list(np.mean(cells[3:], axis=0)),
        # #         **{'labels': ['pre-mature', 'mature', 'slightly-worn', 'severely-worn']})
    print(profit)
