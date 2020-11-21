# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:46:13 2020

@author: jingh
"""

import numpy as np
import copy

from envs.production_system.Buffer import Buffer

class IncomingBuffer(Buffer):

	def __init__(self, template_product):
		super().__init__(np.inf)
		self.template_product = template_product
		self.n_fed = 0

	def initialize(self):
		self.n_fed = 0

	def level(self):
		return 1

	def take(self):
		self.n_fed += 1

		new_product = copy.deepcopy(self.template_product)
		new_product.index = self.n_fed
		return new_product
