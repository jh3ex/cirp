# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:31:47 2020

@author: jingh
"""

import numpy as np


class Product:
	def __init__(self, n_feature, n_process, index):
		"""
		The product object representing a single product

		Parameters
		----------
		n_feature : int
		Number of features for the product.
		n_process : TYPE
		DESCRIPTION.

		Returns
		-------
		None.

		"""
		self.process_parameter = []
		self.processing_time = np.array([0]*n_process)
		self.progress = [False] * n_process

		self.index = index
		self.n_feature = n_feature
		# A product defined by its features

# 	def initialize(self):
# 		# Initialize the product
		self.feature = np.zeros(self.n_feature, dtype=float)

		return

	def is_done(self):
		return np.array(self.progress).all()


	def existing_feature(self):
		return self.feature

	def process(self, stage, process_parameter, processing_time, updated_feature):
		"""
		Using the process parameters to process the product

		Parameters
		----------
		process_id : int
			The index of current process.
		process_parameter : array like
			Process parameters for current process.

		Returns
		-------
		processing_time : float
		The time needed to process this product
		given the process parameters.
		"""
		# Record all past process parameters
		self.process_parameter += process_parameter
		# Record all past processing time
		self.processing_time[stage] =processing_time
		# Record past quality feature
		self.feature = updated_feature


		self.progress[stage] = True





