# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:20:10 2020

@author: jingh
"""

import numpy as np
import copy


class Buffer:
	def __init__(self, cap):
		"""
		Define the object for buffer.

		Parameters
		----------
		cap : int
			Capacity of the buffer.
		is_first : boolean, optional
			If the buffer is first buffer. The default is False.
		is_last : boolean, optional
			If the buffer is last buffer. The default is False.

		Returns
		-------
		None.

		"""
		self.initialize()
		self.cap = cap

		return

	def initialize(self):
		self.content = []
		# Record when the product is put into the buffer
		self.time_stamp = []


	def __is_empty(self):
		return not self.content

	def level(self):
		return len(self.content)

	def vacancy(self):
		return self.cap - self.level()

	def __is_full(self):
		return self.level() == self.cap

	def quote(self):
		if self.__is_empty():
			return np.inf
		else:
			return self.time_stamp[-1]


	def take(self):
		"""
		Try to take product from upstream buffer.

		Returns
		-------
		to_go : tuple(bool, int)
		bool: True if successfully pull a product.
		int: if True, return the index of product.

		"""


		if not self.__is_empty():
			self.content, to_go = self.content[:-1], self.content[-1]
			self.time_stamp = self.time_stamp[:-1]
			return to_go
		else:
			return None


	def put(self, product, time_stamp):
		"""
		Try to put finished product to downstream buffer

		Parameters
		----------
		product : int
			The product that is intended to place in the buffer.
		time_stamp : float
			The time when product is put

		Returns
		-------
		bool
			If the product is successfully put into the buffer.

		"""
		if not self.__is_full():
			# Stack product to the bottom of the buffer
			self.content = [product] + self.content
			self.time_stamp = [time_stamp] + self.time_stamp
			return True

		else:
			return False





class IncomingBuffer(Buffer):

	def __init__(self, template_product, defaulat_level=1):
		super().__init__(np.inf)
		self.template_product = template_product
		self.defaulat_level = defaulat_level
		self.n_fed = 0

	def initialize(self):
		self.n_fed = 0

	def level(self):
		return self.defaulat_level

	def take(self):
		self.n_fed += 1

		new_product = copy.deepcopy(self.template_product)
		new_product.index = self.n_fed
		return new_product



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




class GrindingCB(CompletedBuffer):
	def __init__(self, q_star, default_vacancy=1):
		super().__init__()
		self.q_star = q_star

	def quality(self, product):


		q = sum(product.existing_feature())

		return q < self.q_star



