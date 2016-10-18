import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import func as f


N = 25
x1 = np.random.uniform(-1, 1, N)
x2 = np.random.uniform(-1, 1, N)

#f.WritingInFile(['x1', 'x2'], [x1, x2], 'x1x2.txt')

##for i in range(N):
##    x2[i] = 0.3

#zeros = np.zeros(N)

#y_0_x2 = f.FindResponds(zeros, x2, 'u_y_ej_0_x2.txt', N)

#y_x1_0 = f.FindResponds(x1, zeros, 'u_y_ej_x1_0.txt', N)
matr_X = f.create_X_matr(x1, x2, N)
Y, sigma = f.FindResponds(x1, x2, 'u_y_ej_x1_x2.txt', N)
est_tetta = f.parameter_estimation_tetta(np.float32(matr_X), np.float32(Y))
est_y = f.parameter_estimation_y(est_tetta, np.float32(matr_X))
est_e = f.parameter_estimation_error(np.float32(Y), est_y)
est_sigma_2 = f.parameter_estimation_sigma_2(est_e, N)
check = f.check_adequacy_of_the_model(sigma, est_sigma_2)
intervals, XtX_1 = f.calculate_freq_intervals_for_param(N, est_tetta, est_sigma_2, np.float32(matr_X))
check_1 = f.check_param_importance_of_the_model(est_tetta, est_sigma_2, np.float32(XtX_1))
check_2 = f.check_regres_importance_of_the_model(est_tetta, Y, est_sigma_2, np.float32(matr_X), N)
nu, intervals_for_E = f.calculate_freq_intervals_for_Expected_value(N, est_tetta, est_sigma_2, x1, XtX_1, 1)
est_y, intervals_for_res = f.calculate_freq_intervals_for_response(N, est_tetta, est_sigma_2, x1, XtX_1, 1)
print(check_1)
#f.Graph(x1, y_x1_0)
#f.Graph(x2, y_0_x2)