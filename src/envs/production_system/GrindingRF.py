# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 07:57:46 2020

Grinding machine considering random failures

@author: jingh
"""


from Grinding import Grinding
import numpy as np

# Condider random failure

class GrindingRF(Grinding):
	def __init__(self, MTTR_step, MTBF_step, p1, p2, p3, p4, p5, features, stage, buffer_up, buffer_down, n_product_feature, name=None):
		super().__init__(p1, p2, p3, p4, p5, features, stage, buffer_up, buffer_down, n_product_feature, name)

		self.MTTR_step = MTTR_step
		self.MTBF_step = MTBF_step
		self.w = 0
		pass


	def initialize(self, RandomState):
		Grinding.initialize(self, RandomState)
		self.w = 0


	def processing(self, time_elapsed):

		assert self.status == "processing", "There is not product being processed"

		# Process product
		self.remaining_time -= time_elapsed * (1 - self.w)

		if self.remaining_time <= 0:
			self.status = "to release"

		return self.remaining_time

	def get_node_feature(self):
		node_feature, need_decision = Grinding.get_node_feature(self)

		node_feature.append(self.w)

		return node_feature, need_decision

	def get_feature_size(self):
		return 5

	def tool_check(self):
		if self.w == 0 and self.RD.rand() < 1 / self.MTBF_step:
			# Machine is operational
			self.w = 1
			return

		if self.w == 1 and self.RD.rand() < 1 / self.MTTR_step:
			self.w = 0
			return
