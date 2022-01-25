# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:58:29 2020

@author: jingh
"""

from ProductionGraph import ProductionGraph
from Product import Product
from GrindingRF import GrindingRF
from Buffer import Buffer
from IncomingBuffer import IncomingBuffer
from GrindingCB import GrindingCB


import matplotlib.pyplot as plt
import numpy as np


def process_control(v_range, w_range, a_range):
	"""
	𝑣_1  [0.33, 0.5]"m/s",
	𝑤_1 [28,33] m/s,
	𝑎_1 [0.000012,0.000018] m
	"""
	v = np.random.rand() * (v_range[1] - v_range[0]) + v_range[0]
	w = np.random.rand() * (w_range[1] - w_range[0]) + w_range[0]
	a = np.random.rand() * (a_range[1] - a_range[0]) + a_range[0]

# 	v = np.random.rand() * (0.5 - 0.33) + 0.33
# 	w = np.random.rand() * (33 - 28) + 28
# 	a = np.random.rand() * (0.000018 - 0.000012) + 0.000012

	return [v, w, a]

if __name__ == "__main__":

	seed = 123
	np.random.seed(seed)
	# Define product
	product = Product(n_feature=4, n_process=4, index=0)

	# Process control
# 	v_range, w_range, a_range = [0.33, 0.5], [28, 33], [0.000012, 0.000018]
	v_range, w_range, a_range = [0.2, 0.3], [28, 33], [0.000012, 0.000018]


	# Define incoming buffer
	incoming_buffer = IncomingBuffer(product)

	# Stage 10

	buffer_10_20 = Buffer(cap=10)

	m10_1 = GrindingRF(p1=0.9, p2=1.4, p3=0.5, p4=2, p5=0.01,
				       features=None, stage=0,
					   buffer_up=[incoming_buffer],
					   buffer_down=[buffer_10_20],
					   MTTR_step=50,
					   MTBF_step=100)

	# Stage 20
	buffer_20_30 = Buffer(cap=10)

	m20_1 = GrindingRF(p1=0.85, p2=1.35, p3=0.5, p4=2, p5=0.01,
				       features=None, stage=1,
					   buffer_up=[buffer_10_20],
					   buffer_down=[buffer_20_30],
					   MTTR_step=50,
					   MTBF_step=100)

	# Stage 30
	buffer_30_40 = Buffer(cap=10)

	m30_1 = GrindingRF(p1=0.9, p2=1.5, p3=0.5, p4=2, p5=0.005,
				       features=None, stage=2,
					   buffer_up=[buffer_20_30],
					   buffer_down=[buffer_30_40],
					   MTTR_step=50,
					   MTBF_step=100)

	# Stage 40
	completed_buffer = GrindingCB(q_star=4e-6)

	m40_1 = GrindingRF(p1=0.85, p2=1.3, p3=0.5, p4=2, p5=0.03,
				       features=None, stage=3,
					   buffer_up=[buffer_30_40],
					   buffer_down=[completed_buffer],
					   MTTR_step=50,
					   MTBF_step=100)

	m40_2 = GrindingRF(p1=0.85, p2=1.3, p3=0.5, p4=2, p5=0.03,
				       features=None, stage=3,
					   buffer_up=[buffer_30_40],
					   buffer_down=[completed_buffer],
					   MTTR_step=50,
					   MTBF_step=100)



	machines = [m10_1, m20_1, m30_1, m40_1, m40_2]
	adj_list =[[1], [0, 2], [1, 3], [2], [2]]
	buffers = [incoming_buffer, buffer_10_20, buffer_20_30, buffer_30_40, completed_buffer]


	pg = ProductionGraph(adj_list, machines, buffers, incoming_buffer, completed_buffer)

	pg.initialize(sim_duration=300000, stepsize=10, random_seed=seed)



	n_machine = len(machines)

	parameters = [None] * n_machine

	terminate = False

	output_all = [0]
	yield_all = [0]

	while not terminate:
		pr, output_step, yield_step, terminate = pg.run(parameters)

		output_all.append(output_step + output_all[-1])
		yield_all.append(yield_step + yield_all[-1])

		parameters = [None] * n_machine

		for i, p in enumerate(pr):
			if p is not None:
				parameters[i] = process_control(v_range, w_range, a_range)


	print(pg.get_yield())

	plt.figure(dpi=300)

	plt.plot(yield_all)
	plt.plot([output_all[i] - yield_all[i] for i in range(len(output_all))])
	plt.plot(output_all)
	plt.legend(["Yield", "Defect","Output total"])
	plt.xlabel("time")
	plt.ylabel("production count (parts)")
	plt.show()



