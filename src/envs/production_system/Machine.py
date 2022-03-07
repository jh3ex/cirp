# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 06:51:18 2020

@author: jingh
"""

import numpy as np

class Machine:
	def __init__(self, features, stage, buffer_up, buffer_down, n_product_feature, name=None):

		# Initial machine status
		self.features = features
		self.name = name

		self.stage = stage

		self.buffer_up = buffer_up
		self.buffer_down = buffer_down

		self.n_product_feature = n_product_feature

		return


	def initialize(self):


		self.output = 0
		self.current_product = None
		self.remaining_time = 0
		self.status = "to load"
		self.time = 0


	def quote(self, time_elapsed=0):
		"""
		Quote current status

		Returns
		-------
		str
		Machine status.
		int
		Current product id.

		"""
		if time_elapsed != 0:
			self.time += time_elapsed
		self.tool_check()

		return self.status, self.current_product


	def load(self, product):

		assert self.status == "to load", "Machine is not ready to load"

		# Set current product
		self.current_product = product

		self.status = "awaiting parameter"

		return self.current_product.existing_feature()

	def set_process_parameter(self, process_parameter):

		# Use process model to get cycle time and updated product feature
		processing_time, updated_feature = self.process_model(self.current_product.existing_feature(), process_parameter)

		# Tool model
		# self.tool_model()
		# Update product
		self.current_product.process(self.stage, process_parameter, processing_time, updated_feature)


		self.remaining_time = processing_time

		self.status = "processing"


	def processing(self, time_elapsed):

		assert self.status == "processing", "There is not product being processed"

		# Process product
		self.remaining_time -= time_elapsed

		if self.remaining_time <= 0:
			self.remaining_time = 0.0
			self.status = "to release"

		return self.remaining_time


	def release(self):
		assert self.status == "to release", "Product is not ready to release"
		# Release current product


		self.output += 1

		released_product = self.current_product

		self.current_product = None

		self.status = "to load"

		return released_product

	def need_decision(self):
		return self.status == "awaiting parameter"

	def get_node_feature(self):
		b_up, b_down = 0, 0

		for b in self.buffer_up:
			b_up += b.level()

		for b in self.buffer_down:
			b_down += b.vacancy()

		if self.current_product is not None:
			product_feature = self.current_product.feature.tolist()
		else:
			product_feature = [0.0] * self.n_product_feature

		node_feature = {"stage": self.stage,
						"b_up": b_up,
						"b_down": b_down,
						"remaining_time": self.remaining_time,
						"product_feature": product_feature}


		return node_feature, self.need_decision()

	def get_feature_size(self):
		return 4

	def process_model(self, existing_feature, process_param):
		# Define the process
		pass

	def tool_model(self):
		pass

	def tool_check(self):
		pass






class Grinding(Machine):
	def __init__(self, p1, p2, p3, p4, p5, features, stage, buffer_up, buffer_down, n_product_feature, name=None):
		super().__init__(features, stage, buffer_up, buffer_down, n_product_feature, name)
		self.p1 = p1
		self.p2 = p2
		self.p3 = p3
		self.p4 = p4
		self.p5 = p5

		return


	def process_model(self, existing_feature, process_param):

		v, w, a = process_param


		loc = (v * a / w)**self.p1
		scale = a**self.p2 * w**self.p3 * v**self.p4

		existing_feature[self.stage] = np.random.normal(loc=loc, scale=scale)

		processing_time = self.p5 / (v*a)

		return processing_time, existing_feature



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
				self.w = 0
				self.time_when_fail = self.time + np.random.exponential(self.MTBF)




class GrindingRBE(Grinding):
	def __init__(self, tp, ep, p6, time_to_dress, fixed_dress_schedule, pass_to_dress,
				 p1, p2, p3, p4, p5, p5_scale, features, stage, buffer_up, buffer_down, n_product_feature, name=None):
		super().__init__(p1, p2, p3, p4, p5, features, stage, buffer_up, buffer_down, n_product_feature, name)

		self.tp = np.array(tp)
		self.n_tool_state = self.tp.shape[0]

		for i in range(self.n_tool_state):
			self.tp[i] /= self.tp[i].sum()

		self.ep = np.array(ep)

		for i in range(self.n_tool_state):
			self.ep[i] /= self.ep[i].sum()

		self.p6 = p6
		self.p5_scale = p5_scale

		self.time_to_dress = time_to_dress
		self.fixed_dress_schedule = fixed_dress_schedule

		if self.fixed_dress_schedule:
			self.pass_to_dress = pass_to_dress


		self.w = 0
		pass


	def quote(self, time_elapsed=0):

		if time_elapsed != 0:
			self.time += time_elapsed
		self.tool_check(time_elapsed)

		return self.status, self.current_product


	def initialize(self):
		self.tool_state = np.random.choice(self.n_tool_state)
		self.tool_ob = np.random.choice(self.n_tool_state, p=self.ep[self.tool_state])

		self.tool_belief = np.random.rand(self.n_tool_state)
		self.tool_belief /= self.tool_belief.sum()



		self.passes = 0
		Grinding.initialize(self)


	def load(self, product):

		assert self.status == "to load", "Machine is not ready to load"

		# Set current product
		self.current_product = product

		self.status = "awaiting parameter"

		self.tool_state = np.random.choice(self.n_tool_state, p=self.tp[self.tool_state])
		self.tool_ob = np.random.choice(self.n_tool_state, p=self.ep[self.tool_state])

		# One step prediciton
		self.tool_belief = (self.tp * self.tool_belief.reshape(4, 1)).sum(axis=0)
		# One step correction
		self.tool_belief = self.tool_belief * self.ep[:, self.tool_ob]
		self.tool_belief /= self.tool_belief.sum()



		return self.current_product.existing_feature()


	def process_model(self, existing_feature, process_param):

		v, w, a = process_param

		loc = (v * a / w)**self.p1 * self.p6[self.tool_state]


		scale = a**self.p2 * w**self.p3 * v**self.p4

		existing_feature[self.stage] = np.random.normal(loc=loc, scale=scale)

		processing_time = self.p5_scale[self.tool_state] * self.p5 / (v*a)

		return processing_time, existing_feature

	def processing(self, time_elapsed):

		assert self.status == "processing", "There is not product being processed"

		# Process product
		self.remaining_time -= time_elapsed * (1 - self.w)

		if self.remaining_time <= 0:
			self.remaining_time = 0.0
			self.status = "to release"
			self.passes += 1

		return self.remaining_time


	def dress(self):
		assert self.status == "to dress", "Machine is not ready to dress"
		self.dress_time = self.time_to_dress
		self.status = "dressing"
		self.w = 1
		self.tool_state = 0
		self.tool_belief = np.zeros(self.n_tool_state)
		self.tool_belief[0] = 1
		self.passes = 0

	def release(self):
		assert self.status == "to release", "Product is not ready to release"
		# Release current product


		self.output += 1

		released_product = self.current_product

		self.current_product = None

		self.status = "to dress"

		return released_product


	def get_node_feature(self):
		node_feature, need_decision = Grinding.get_node_feature(self)

		# Append machine random failure status
		node_feature["w"] = self.w
		# Append tool state observation

		node_feature["tool_ob"] = self.tool_ob
		node_feature["tool_belief"] = self.tool_belief.tolist()
		# node_feature["passes"] = self.passes

		need_dress = self.status == "to dress"

		return node_feature, need_decision, need_dress

	def get_feature_size(self):
		return 6

	def tool_check(self, time_elapsed):
		if self.fixed_dress_schedule and self.status == "to dress":
			if self.passes >= self.pass_to_dress:
				self.dress()
			else:
				self.status = "to load"

		# if self.w == 0:
		# 	self.tool_state = np.random.choice(self.n_tool_state, p=self.tp[self.tool_state])

		if self.w == 1:
			self.dress_time -= time_elapsed
			if self.dress_time <= 0:
				self.w = 0
				self.status = "to load"
