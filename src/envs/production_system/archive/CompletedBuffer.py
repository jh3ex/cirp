# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 21:23:44 2020

@author: jingh
"""


import numpy as np

from envs.production_system.Buffer import Buffer

class CompletedBuffer(Buffer):

	def __init__(self, default_vacancy=1):
		super().__init__(np.inf)
		self.default_vacancy = default_vacancy
		self.initialize()

	def initialize(self):
		self.output = 0.0
		self.final_yield = 0.0

	def put(self, product, time_stamp):
		self.output += 1.0
		if self.quality(product):
			self.final_yield += 1.0
		return True

	def level(self):
		return 0

	def vacancy(self):
		return self.default_vacancy

	def output_and_yield(self):
		return self.output, self.final_yield

	def quality(self, product) -> bool:
		"""
		Quality standard used to evaluate final product.

		Parameters
		----------
		product : object
			Final product.

		Returns
		-------
		bool
			If the product pass quality inspection.

		"""
		pass
