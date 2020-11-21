# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:48:12 2020

@author: jingh
"""

from envs.production_system.CompletedBuffer import CompletedBuffer


class GrindingCB(CompletedBuffer):
	def __init__(self, q_star):
		super().__init__()
		self.q_star = q_star

	def quality(self, product):


		q = sum(product.existing_feature())

		return q < self.q_star
