import time
import numpy as np
import random
import math
from absl import logging
from .scenes import get_map_params
from functools import partial
import math



class Vehicle():
    def __init__(self, id, x_init, y_init, x_goal, y_goal, **mid_xys):
        self.id = id
        self.x_init = x_init
        self.y_init = y_init
        self.mid_points_dict = mid_xys

    def gas(self):
        assert self.x_mid == self.x_mid or self.y_mid

    def brake(self):
        # do nothing here
        return


class TrafficJunction:
    def __init__(self, difficulty='easy'):
        self._init_board(difficulty)


    def _init_board(self, difficulty):
        if difficulty == "easy":
            #N, W
            self.init_pos_dict = {'W': (0, 3), 'N': (3, 0)}
            self.goal_pos_dict = {'E': (6, 3), 'S': (3, 6)}
            self.mid_point_dict = {'WE': [], 'NS': []}
        else:
            #NW, NE, WN, WS, EN, ES, SW, SE
            self.init_pos_dict = {'NW': (0, 4), 'NE':(0, 12),
                                  'WN': (5, 0), 'WS': (13, 0),
                                  'EN': (4, 17), 'ES': (12, 17),
                                  'SW': (17, 5), 'SE': (17, 13)}
            self.goal_pos_dict = {'NW': (0, 5), 'NE':(0, 13),
                                  'WN': (4, 0), 'WS': (12, 0),
                                  'EN': (5, 17), 'ES': (13, 17),
                                  'SW': (17, 4), 'SE': (17, 12)}

            self.mid_point_dict = {'NWNE': [(5, 4), (5, 13)], 'NWWN': [(4, 4)], 'NWWS': [(12, 4)], 'NWEN': [(5, 4)], 'NWES': [(13, 4)], 'NWSW': [], 'NWSE':[(13, 4), (13, 12)],
                                   'NENW': [(4, 12), (4, 5)], 'NEWN': [(4, 12)], 'NEWS': [(12, 12)], 'NEEN': [(5, 12)], 'NEES': [(13, 12)], 'NESW': [(12, 12), (12, 4)], 'NESE':[],
                                   'WNNW': [(5, 5)], 'WNWE': [(5, 13)], 'WNWS': [(5, 4), (12, 4)], 'WNEN': [], 'WNES': [(5, 12), (13, 12)], 'WNSW': [(5, 4)], 'WNSE':[(5, 12)],
                                   'WSNW': [(13, 5)], 'WSNE': [(13, 13)], 'WSWN': [(13, 5), (4, 5)], 'WSEN': [(13, 13), (5, 13)], 'WSES': [], 'WSSW': [(13, 4)], 'WSSE':[(13, 12)],
                                   'ENNW': [(4, 5)], 'ENNE': [(4, 13)], 'ENWN': [], 'ENWS': [(4, 4), (12, 4)], 'ENES': [(4, 12), (13, 12)], 'ENSW': [(4, 4)], 'ENSE':[(4, 12)],
                                   'ESNW': [(12, 5)], 'ESNE': [(12, 13)], 'ESWS': [], 'ESWN': [(12, 5), (4, 5)], 'ESEN': [(12, 13), (5, 13)],  'ESSW': [(12, 4)], 'NESE':[(12, 12)],
                                   'SWNW': [], 'SWNE': [(5, 5), (5, 13)], 'SWWS': [(12, 5)], 'SWWN': [(4, 5)], 'SWEN': [(5, 5)], 'SWES': [(13, 5)], 'SWSE':[(13, 5), (13, 12)],
                                   'SENW': [(4, 13), (4, 5)], 'SENE': [], 'SEWS': [(12, 13)], 'SEWN': [(4, 13)], 'SEEN': [(5, 13)], 'SEES': [(13, 13)], 'SESW':[(12, 13),(12, 4)]}



    def _add_vehicle(self):
