# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 06:51:18 2020

@author: jingh
"""



class Machine:
	def __init__(self, features, stage, buffer_up, buffer_down):

		# Initial machine status
		self.features = features


		self.stage = stage

		self.buffer_up = buffer_up
		self.buffer_down = buffer_down

		return


	def initialize(self, RandomState):

		self.RD = RandomState
		self.output = 0
		self.current_product = None
		self.remaining_time = 0
		self.status = "to load"



# 	def is_ready_to_release(self):
# 		# True if current product is completed
# 		return (self.remaining_time <= 0) and (not self.is_ready_to_load())


# 	def is_processing(self):
# 		return self.remaining_time > 0


# 	def is_ready_to_load(self):
# 		# True if machine is available for taking new product
# 		return not self.current_product

# 	def is_ready_to_process(self):
# 		return self.ready_to_process
# 		# return self.remaining_time <= 0 and self.current_product

	def quote(self):
		"""
		Quote current status

		Returns
		-------
		str
		Machine status.
		int
		Current product id.

		"""
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
		self.tool_model()
		# Update product
		self.current_product.process(self.stage, process_parameter, processing_time, updated_feature)


		self.remaining_time = processing_time

		self.status = "processing"
		pass

	def processing(self, time_elapsed):

		assert self.status == "processing", "There is not product being processed"

		# Process product
		self.remaining_time -= time_elapsed

		if self.remaining_time <= 0:
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


	def node_feature(self):
		b_up, b_down = 0, 0

		for b in self.buffer_up:
			b_up += b.level()

		for b in self.buffer_down:
			b_down += b.vacancy()

		return [b_up, b_down, self.remaining_time, self.current_product.feature], self.is_ready_to_process()


	def process_model(self, existing_feature, process_param):
		# Define the process
		pass

	def tool_model(self):
		pass

	def tool_check(self):
		pass





