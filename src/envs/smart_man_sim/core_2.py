import time
import numpy as np
import random
import math
from absl import logging
from .scenes import get_map_params

"""
Machines only have two actions in this batch of experiments: run, maintain
"""



class Transition():
    """
    machine health states transition rules:
    -There are 4 health states: pre-mature, mature, slightly damaged, severely damaged
    """
    def __init__(self, tran_matrix, length=40, schedule=False):
        self._init_matrix(tran_matrix)
        self.schedule = schedule
        self.length = length
        if schedule:
            self.init = np.array([[[1., 0, 0, 0],
                                   [0, 1., 0, 0],
                                   [0, 0, 1., 0],
                                   [0, 0, 0, 1.]],
                                  [[1, 0, 0, 0],
                                   [1, 0, 0, 0],
                                   [1, 0, 0, 0],
                                   [1, 0, 0, 0]]])
            self.decay = (self.init - self.T) / length

    def _init_matrix(self, tran_matrix):
        """
        T = [[[], [], [], []],
             [[], [], [], []],
             [[], [], [], []]]
        """
        self.T = tran_matrix
        assert len(self.T.shape) == 3 and self.T.shape[0] == 2 and self.T.shape[1] == 4 and self.T.shape[2] == 4

    def transit(self, init_state, action, steps=None):
        if not self.schedule:
            T_a = self.T[action]
            p = T_a[init_state]
            next_state = np.random.choice(4, 1, p=p)[0]
            return next_state
        else:
            if steps[init_state] > self.length:
                steps = self.length
            T = self.init - self.decay * steps
            T_a = T[action]
            p = T_a[init_state] + 1e-9
            p /= np.sum(p)
            next_state = np.random.choice(4, 1, p=p)[0]
            return next_state

ACTION = {0: 1000,
          1: 0,
          2: (12, 3)}

class Machine():
    def __init__(self, id, T, cell_id, config):
        self.T = T
        self.id = id
        self.cell_id = cell_id
        self.init_h_state()
        self.under_m = False
        self.deque = 0
        self.anti_jobs = 0
        self.action = None
        self.m_cd = 0
        self.ACTION = config['actions']
        self.COST = config['costs']
        self.h_history = np.zeros(4)
        self.restart = 0
        self.breakdown = False

    def step(self, action):
        if action == 0:
            assert self.h_state != 3
            self.action = action
            n = self.ACTION[self.action]
            self.request_parts(n)
            self.state_time[self.h_state] += 1
        elif action == 1:
            self.action = action
            self.request_parts(0)
            self.register_m()
            self.state_time[self.h_state] += 1
        else:
            raise ValueError('action [%d] is out of range'%action)
        self.h_history[self.h_state] += 1

    def init_h_state(self, random=False):
        self.h_state = 0
        self.state_time = [0] * 4

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
        self.request_jobs = 0
        if self.m_cd == 0:
            self.m_cd = math.floor(np.random.normal(12, 3))

    def proceed(self):
        if not self.under_m:
            self.deque = self.anti_jobs
            self.anti_jobs = 0
            self.request_jobs = 0
            new_h_state = self.T.transit(self.h_state, self.action, self.state_time)
            self.h_state = new_h_state
        else:
            self.m_cd -= 1
            self.deque = self.anti_jobs
            if self.m_cd == 0:
                self.under_m = False
                new_h_state = self.T.transit(self.h_state, self.action, self.state_time)
                self.h_state = new_h_state
                self.state_time = [0] * 4
                self.restart += 1

    @property
    def valid_actions(self):
        if self.h_state == 3 or self.under_m:
            valid_actions = [0., 1.]
        else:
            valid_actions = [1] * 2
        return valid_actions

    @property
    def cost(self):
        if self.action == 0 and self.h_state != 3:
            cost = self.COST[0]
            # print('Agent {} took action {} and incur {} cost'.format(self.id, self.action, cost))
        elif self.h_state == 3 and self.state_time[3] == 0:
            cost = self.COST[-1]
            if self.action == 0:
                cost += self.COST[0]
                # print('Agent {} took action {}, and breakdown, and incur {} cost'.format(self.id, self.action, cost))
            else:
                raise ValueError('self.action cannot take {}'.format(self.action))
        elif self.action == 1:
            cost = self.COST[1]
            # print('Agent {} took action {} and incur {} cost'.format(self.id, self.action, cost))
        else:
            raise ValueError("Our agent going to wrong state, action pair {}{}".format(self.h_state, self.action))
        return cost


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



class Simulation():
    def __init__(self, map_name):
        self._initialize(map_name)

    def _initialize(self, map_name):
        config = get_map_params(map_name)
        cell_ids = config['cells']
        machine_ids = config['machines']
        transitions = config['transitions']
        self.sale_price = config['sale_price']

        machine_ids = np.array(machine_ids).reshape([len(cell_ids), -1])
        self.machines = []
        self.cells = []
        for i in range(machine_ids.shape[0]):
            cell_id = i
            self.cells.append(Cell(cell_id))
            transition = transitions[i]
            T = Transition(transition, schedule=False)
            for j in range(machine_ids.shape[1]):
                machine_id = machine_ids[i, j]
                self.machines.append(Machine(machine_id, T, cell_id, config))
            self.cells[-1].add_machines(self.machines[-machine_ids.shape[1]:])
            if i > 0:
                p_cell = self.cells[i-1]
                self.cells[-1].add_cells(p_cell)

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
            following_buffer = self.cells[cell+1].buffer_size
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
    map_name = 226
    sim = Simulation(map_name)
    sim.step([0]*6)
    num = 0
    profit = 0
    while num < 128:
        actions = []
        for i in range(6):
            action_p = np.array([1., 0.])
            valid_actions = sim.get_avail_agent_actions(i)
            # print('valid_action is {}'.format(valid_actions))
            action_p = np.array(valid_actions, dtype=np.float32) * (action_p + 1e-9)
            p = action_p / np.sum(action_p)
            # print('p is {}'.format(p))
            action = np.random.choice(2, 1, p=p)[0]
            actions.append(action)
        sim.step(actions)
        print("Actions are {}".format(actions))
        print("States are {}".format([machine.h_state for machine in sim.machines]))
        num += 1
        print(sim.profit)
        profit += sim.profit
    cells = []
    for m in sim.machines:
        cells.append(m.h_history/m.restart)
        print("States history are {}".format(m.h_history/m.restart))
    cells = np.asarray(cells)
    print(np.mean(cells[:3], axis=0))
    #pie_plot(*list(np.mean(cells[:3], axis=0)),
    #         **{'labels': ['pre-mature', 'mature', 'slightly-worn', 'severely-worn']})
    print(np.mean(cells[3:], axis=0))
    #pie_plot(*list(np.mean(cells[3:], axis=0)),
    #         **{'labels': ['pre-mature', 'mature', 'slightly-worn', 'severely-worn']})
    print(profit)
