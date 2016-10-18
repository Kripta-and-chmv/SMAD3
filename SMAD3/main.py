import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import func as f


N = 25
x1, x2 = f.get_x1_x2('x1x2.txt')
Y = np.array(f.get_y('u_y_ej_x1_x2.txt'))
Y1 = np.array(f.get_y('u_y_ej_x1_0.txt'))
matr_X = f.create_X_matr(x1, x2, N)
est_tetta = f.parameter_estimation_tetta(matr_X, Y)
est_y = f.parameter_estimation_y(est_tetta, matr_X)
est_e = f.parameter_estimation_error(Y, est_y)
est_sigma_2 = f.parameter_estimation_sigma_2(est_e, N)
intervals, XtX_1 = f.calculate_freq_intervals_for_param(N, est_tetta, est_sigma_2, matr_X)
check_1 = f.check_param_importance_of_the_model(est_tetta, est_sigma_2, XtX_1)
check_2 = f.check_regres_importance_of_the_model(est_tetta, Y, est_sigma_2, matr_X, N)
nu, intervals_for_E, nu_real = f.calculate_freq_intervals_for_Expected_value(N, est_tetta, est_sigma_2, x1,  XtX_1, 1)
est_y1, intervals_for_res = f.calculate_freq_intervals_for_response(N, est_tetta, est_sigma_2, x1,  XtX_1, 1)
f.Graph(x1, intervals_for_E, nu)
f.Graph(x1, intervals_for_res, est_y1)
