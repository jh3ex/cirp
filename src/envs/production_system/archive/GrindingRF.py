# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 07:57:46 2020

Grinding machine considering random failures

@author: jingh
"""


from envs.production_system.Grinding import Grinding
import numpy as np

# Condider random failure

class GrindingRF(Grinding):
	def __init__(self, MTTR, MTBF, p1, p2, p3, p4, p5, features, stage, buffer_up, buffer_down, n_product_feature, name=None):
		super().__init__(p1, p2, p3, p4, p5, features, stage, buffer_up, buffer_down, n_product_feature, name)

		self.MTTR = MTTR
		self.MTBF = MTBF
		self.w = 0
		pass


	def initialize(self):
		Grinding.initialize(self)
		self.w = 0
		self.time_when_fail = np.random.exponential(self.MTBF)



	def processing(self, time_elapsed):

		assert self.status == "processing", "There is not product being processed"



		# Process product
		self.remaining_time -= time_elapsed * (1 - self.w)

		if self.remaining_time <= 0:
			self.remaining_time = 0.0
			self.status = "to release"

		return self.remaining_time

	def get_node_feature(self):
		node_feature, need_decision = Grinding.get_node_feature(self)

		# Append machine random failure status
		node_feature["w"] = self.w

		return node_feature, need_decision

	def get_feature_size(self):
		return 5

	def tool_check(self):
		if self.w == 0:
			if self.time >= self.time_when_fail:
				# Machine will fail
				self.w = 1
				self.time_when_back = self.time + np.random.exponential(self.MTTR)
		else:
			if self.time >= self.time_when_back:
				# Machine comes back
				self.w = 0
				self.time_when_fail = self.time + np.random.exponential(self.MTBF)
