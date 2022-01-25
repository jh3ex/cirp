# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 07:04:31 2020

@author: jingh
"""


import numpy as np
import pandas as pd

from Buffer import Buffer
from Product import Product
from Machine import Machine

class ProductionGraph:
	def __init__(self, adj_list, machines, buffers, incoming_buffer, completed_buffer):
		"""
		Construct a graph for production system

		Parameters
		----------
		adj_list : list like
		Adjacency list.
		buffer_cap : list like or array
		Buffer capacity for each machine, null for end-of-line machine
				stepsize : float
					Stepsize for simulation

		Returns
		-------
		None.

		"""
		self.adj_list = adj_list

		self.n_machine = len(machines)

		# Generate all matrices
		self.__matrices()
		self.machines = machines
		self.buffers = buffers
		self.incoming_buffer = incoming_buffer
		self.completed_buffer = completed_buffer




	def __matrices(self):
		"""
		Create adjacency matrix

		Returns
		-------
		adj_matrix: Adjacency matrix
		deg_matrix: Degree matrix

		"""
		self.adj_matrix = np.zeros((self.n_machine, self.n_machine), dtype=int)
		self.deg_matrix = np.zeros((self.n_machine, self.n_machine), dtype=int)

		for i in range(self.n_machine):
			self.adj_matrix[i, self.adj_list[i]] = 1
			self.deg_matrix[i, i] = len(self.adj_list[i])

		return


	def adjacency_matrix(self):
		return self.adj_matrix


	def degree_matrix(self):
		return self.deg_matrix


	def initialize(self, sim_duration, stepsize, random_seed):
		"""
		Set the initial conditions

		Parameters
		----------
		sim_duration : float
		Simulation duration.
		random_seed : int
		random seed for simulation.

		Returns
		-------
		None.

		"""
		# Random seed for simulation
		self.RD = np.random.RandomState(seed=random_seed)

		# Simulation time horizon
		self.sim_duration = sim_duration
		self.stepsize = stepsize
		# Total steps given stepsize
		self.total_step = int(self.sim_duration / self.stepsize)


		# Initialize everthing
		for m in self.machines:
			m.initialize(self.RD)

		for b in self.buffers:
			b.initialize()

		self.incoming_buffer.initialize()
		self.completed_buffer.initialize()

		self.time = 0
		self.terminate = False

	def run(self, parameters):
		"""
		Run the system for one time step (stepsize)

		Returns
		-------
		None.

		"""
		assert not self.terminate, "The simulation is terminated"


		self.time += self.stepsize

		output_before, yield_before = self.completed_buffer.output_and_yield()

		parameter_request = [None] * self.n_machine

		for i, m in enumerate(self.machines):
			# Iterates over all machines
			status, product = m.quote()
			if status == "processing":
				m.processing(self.stepsize)
			elif status == "to release":
				for b in m.buffer_down:
					if b.put(product, self.time):
						m.release()
						break
			elif status == "to load":
				for b in m.buffer_up:
					product = b.take()
					if product is not None:
						existing_feature = m.load(product)
						parameter_request[i] = existing_feature
						break
			elif status == "awaiting parameter":
				m.set_process_parameter(parameters[i])

		output_after, yield_after = self.completed_buffer.output_and_yield()
		output_step = output_after - output_before
		yield_step = yield_after - yield_before

		self.terminate = (self.time >= self.sim_duration)

		return parameter_request, output_step, yield_step, self.terminate


	def get_node_feature(self, i):
		return self.machines[i].get_node_feature()

	def get_yield(self):
		return self.completed_buffer.output_and_yield()




if __name__ == "__main__":

	pass


