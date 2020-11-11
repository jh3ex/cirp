# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:34:34 2020

@author: jingh
"""


from Machine import Machine
import numpy as np

class Grinding(Machine):
	def __init__(self, p1, p2, p3, p4, p5, features, stage, buffer_up, buffer_down):
		super().__init__(features, stage, buffer_up, buffer_down)
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

		existing_feature[self.stage] = self.RD.normal(loc=loc, scale=scale)

		processing_time = self.p5 / (v*a)

		return processing_time, existing_feature



