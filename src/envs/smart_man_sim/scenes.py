import numpy as np
"""
buffer_size: 48
limit: 48

buffer_size: 32
limit: 48

buffer_size: 32
limit: 128
"""

map_param_registry = {}

def get_map_params(map_name):
    return map_param_registry[map_name]

map_param_registry = {226:{'cells': [0, 1],
                            'machines': [0, 1, 2, 3, 4, 5],
                            'actions': [1000, 0], #action 0, 1, 2
                            'costs': [100, 50, 2000], #cost of running, cost of not working, cost of maintaining, cost of system breaking down}
                            'n_agents': 6,
                            'n_cells': 2,
                            'n_actions': 2,
                            'limit': 128,
                            'sale_price':.5,
                            'continuous_trans': False,
                            'transitions': [np.array([[[.7, .29, .009, .001],
                                                       [0, .9, .09, .01],
                                                       [0, 0, .6, .4],
                                                       [0, 0, 0, 1]],
                                                      [[1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0]]]),
                                            np.array([[[.65, .34, .009, .001],
                                                       [0, .9, .099, .001],
                                                       [0, 0, .55, .45],
                                                       [0, 0, 0, 1]],
                                                      [[1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0]]])]},
                      26:{'cells': [0, 1],
                            'machines': [0, 1, 2, 3, 4, 5],
                            'actions': [1000, 0, 0], #action 0, 1, 2
                            'costs': [100, 80, 50, 2000], #cost of running, cost of not working, cost of maintaining, cost of system breaking down}
                            'n_agents': 6,
                            'n_cells': 2,
                            'n_actions': 3,
                            'limit': 128,
                            'sale_price':.5,
                            'continuous_trans': False,
                            'transitions': [np.array([[[.7, .29, .009, .001],
                                                       [0, .95, .049, .001],
                                                       [0, 0, .6, .4],
                                                       [0, 0, 0, 1]],
                                                      [[.8, .19, .009, .001],
                                                       [0, 0.98, .019, .001],
                                                       [0, 0, 0.8, .2],
                                                       [0, 0, 0, 0]],
                                                      [[1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0]]]),
                                            np.array([[[.65, .34, .009, .001],
                                                       [0, .9, .099, .001],
                                                       [0, 0, .55, .45],
                                                       [0, 0, 0, 1]],
                                                      [[.75, .24, .009, .001],
                                                       [0, 0.93, .069, .001],
                                                       [0, 0, 0.75, .25],
                                                       [0, 0, 0, 0]],
                                                      [[1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0]]])]},

                    

                      261:{'cells': [0, 1],
                           'machines': [0, 1, 2, 3, 4, 5],
                           'actions': [1000, 0, 0], #action 0, 1, 2
                           'costs': [100, 40, 65, 2000], #cost of running, cost of not working, cost of maintaining, cost of system breaking down}
                           'n_agents': 6,
                           'n_cells': 2,
                           'n_actions': 3,
                           'limit': 48,
                           'sale_price':.5,
                           'continuous_trans': True,
                           'dist': 'gamma',
                           'first_params': [4, 12, 4, 11],
                           'second_params':[1, 1, 1, 0.5],
                           'lower_bounds':[1, 1, 1, 9]},

                    262:{'cells': [0, 1],
                         'machines': [0, 1, 2, 3, 4, 5],
                         'actions': [1000, 0, 0], #action 0, 1, 2
                         'costs': [100, 40, 65, 2000], #cost of running, cost of not working, cost of maintaining, cost of system breaking down}
                         'n_agents': 6,
                         'n_cells': 2,
                         'n_actions': 3,
                         'limit': 48,
                         'sale_price':.5,
                         'continuous_trans': True,
                         'dist': 'gamma',
                         'first_params': [8, 24, 8, 11],
                         'second_params':[.5, .5, .5, 0.5],
                         'lower_bounds':[1, 1, 1, 9]},

                     263:{'cells': [0, 1],
                          'machines': [0, 1, 2, 3, 4, 5],
                          'actions': [1000, 0, 0], #action 0, 1, 2
                          'costs': [100, 40, 65, 2000], #cost of running, cost of not working, cost of maintaining, cost of system breaking down}
                          'n_agents': 6,
                          'n_cells': 2,
                          'n_actions': 3,
                          'limit': 48,
                          'sale_price':.5,
                          'continuous_trans': True,
                          'dist': 'static',
                          'first_params': [4, 12, 4, 11],
                          'second_params':[1, 1, 1, 0.5],
                          'lower_bounds':[1, 1, 1, 9]},

                    361:{'cells': [0, 1, 2],
                         'machines': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                         'actions': [1000, 0, 0], #action 0, 1, 2
                         'costs': [100, 40, 65, 2000], #cost of running, cost of not working, cost of maintaining, cost of system breaking down}
                         'n_agents': 9,
                         'n_cells': 3,
                         'n_actions': 3,
                         'limit': 48,
                         'sale_price':.5,
                         'continuous_trans': True,
                         'dist': 'gamma',
                         'first_params': [4, 2, 4, 11],
                         'second_params':[1, 6, 1, 0.5],
                         'lower_bounds':[1, 1, 1, 9]},



                      39:{'cells': list(range(3)),
                            'machines': list(range(9)),
                            'actions': [1000, 0, 0], #action 0, 1, 2
                            'costs': [100, 50, 50, 1000], #cost of running, cost of not working, cost of maintaining, cost of system breaking down}
                            'limit': 48,
                            'n_agents': 9,
                            'n_actions': 3,
                            'n_cells': 3,
                            'sale_price':.9,
                            'transitions': [np.array([[[.7, .29, .009, .001],
                                                       [0, .95, .049, .001],
                                                       [0, 0, .6, .4],
                                                       [0, 0, 0, 1]],
                                                      [[.8, .19, .009, .001],
                                                       [0, 0.98, .019, .001],
                                                       [0, 0, 0.8, .2],
                                                       [0, 0, 0, 0]],
                                                      [[1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0]]]),
                                            np.array([[[.7, .29, .009, .001],
                                                                       [0, .95, .049, .001],
                                                                       [0, 0, .6, .4],
                                                                       [0, 0, 0, 1]],
                                                                      [[.8, .19, .009, .001],
                                                                       [0, 0.98, .019, .001],
                                                                       [0, 0, 0.8, .2],
                                                                       [0, 0, 0, 0]],
                                                                      [[1, 0, 0, 0],
                                                                       [1, 0, 0, 0],
                                                                       [1, 0, 0, 0],
                                                                       [1, 0, 0, 0]]]),
                                            np.array([[[.65, .34, .009, .001],
                                                       [0, .9, .099, .001],
                                                       [0, 0, .55, .45],
                                                       [0, 0, 0, 1]],
                                                      [[.75, .24, .009, .001],
                                                       [0, 0.93, .069, .001],
                                                       [0, 0, 0.75, .25],
                                                       [0, 0, 0, 0]],
                                                      [[1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0],
                                                       [1, 0, 0, 0]]])]}}

if __name__ == '__main__':
    print(get_map_params(26))
