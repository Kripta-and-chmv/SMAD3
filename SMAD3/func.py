import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math
import scipy.stats as st

def func_2(x):
    return x

def func_3(x):
    return sp.exp(-x ** 2)

def func_4(x):
    return x ** 2

def func_5(x):
    return x ** 4

def create_X_matr(x1, x2, N):
    X = []
    for i in range(N):
        X.append([])
        X[i].append(1.)
        X[i].append(func_2(x1[i]))
        X[i].append(func_3(x1[i]))
        X[i].append(func_4(x2[i]))
        X[i].append(func_5(x2[i]))
    return np.array(X)

def parameter_estimation_tetta(matr_X, Y):
    XtX = np.matmul(matr_X.T, matr_X)
    XtX_1 = np.linalg.inv(XtX)
    XtX_1_Xt = np.matmul(XtX_1, matr_X.T)
    est_tetta = np.matmul(XtX_1_Xt, Y)
    return est_tetta

def parameter_estimation_y(est_tetta, matr_X):
    est_y = np.matmul(matr_X, est_tetta)
    return est_y

def parameter_estimation_error(Y, est_y):
    est_e = Y - est_y
    return est_e

def parameter_estimation_sigma_2(est_e, N):
    est_sigma_2 = np.matmul(est_e.T, est_e) / (N - 4)
    return est_sigma_2

def check_adequacy_of_the_model(sigma, est_sigma_2):
    Ft = 1.5705
    F = est_sigma_2 / sigma ** 2
    ##########
    #???? ??????
    #alpha = 0.05
    #p_value = st.f.cdf(F, 21, float('Inf'))
    #if p_value > alpha:
    #    return False
    #else:
    #    return True
        
    if F <= Ft:
        #�� �����������
        return True
    else:
        return False

#################################
def calculate_freq_intervals_for_param(N, est_tetta, est_sigma_2, matr_X):
    #������������� ��������� ��� ������� ��������� ������ ���������
    alpha = 0.05
    est_sigma = math.sqrt(est_sigma_2)
    freq_intervals = [[], []]
    for i in range(5):
       delta = st.t.ppf(alpha / 2, N - 5)
       XtX_1 = np.linalg.inv(np.matmul(matr_X.T, matr_X))
       delta = est_sigma * XtX_1[i][i]
       freq_intervals[0].append(est_tetta[i] - delta)
       freq_intervals[1].append(est_tetta[i] + delta)
    return freq_intervals, XtX_1


def check_param_importance_of_the_model(est_tetta, est_sigma_2, XtX_1):
    #�������� �������� ���������� ������� ��������� ������
    Ft = 4.3512
    F = []
    check = []
    for i in range(5):
        F.append((est_tetta[i] ** 2 )/ (est_sigma_2 * XtX_1[i][i]))
    
        if F[i] < Ft:
        #�� �����������
            k = True
        else:
            k = False
        check.append(k)
    return check

def check_regres_importance_of_the_model(est_tetta, Y, matr_X, N):
    #�������� �������� � ���������� ��������� ���������
    summ = .0
    YtY =  np.matmul(Y.T, Y)
    est_tet_t_XtY =  np.matmul( np.matmul(est_tetta.T, matr_X.T), Y)
    RSS = YtY - est_tet_t_XtY
    for i in range(N):
        summ += Y[i]
    MY = summ / N
    RSS_H = YtY - N * MY **2
    q = 4
    F = ((RSS_H - RSS) / q) / (RSS / (N - 5))
    Ft = 2.8661
    if F < Ft:
        #�� �����������
        return True
    else:
        return False

def calculate_freq_intervals_for_Expected_value(N,est_tetta, est_sigma_2, x, XtX_1, flag):
    est_sigma = math.sqrt(est_sigma_2)
    est_tett = np.array(est_tetta)
    freq_intervals = [[], []]
    alpha = 0.05
    if flag == 1:
        f1 = []
        for i in range(N):
            f1.append([])
            f1[i].append(1.)
            f1[i].append(func_2(x[i]))
            f1[i].append(func_3(x[i]))
            f1[i].append(.0)
            f1[i].append(.0)
        f1 = np.float32(np.array(f1))
        nu = np.matmul(f1.T, est_tett)
        tmp = np.matmul(np.matmul(f1, XtX_1), f1.T)
        delta = est_sigma * math.sqrt(tmp)
        delta *= st.t.ppf(alpha / 2, N - 5)
        for i in range(5):
            freq_intervals[0].append(nu[i] - delta)
            freq_intervals[1].append(nu[i] + delta)
    else:
        f2 = []
        for i in range(N):
            f2.append([])
            f2[i].append(.0)
            f2[i].append(.0)
            f2[i].append(.0)
            f2[i].append(func_4(x[i]))
            f2[i].append(func_5(x[i]))
        f2 = np.array(f2)
        nu = np.matmul(f2.T,  est_tetta)
        delta = est_sigma * math.sqrt(np.matmul(np.matmul(f2.T, XtX_1), f2))
        delta *= st.t.ppf(alpha / 2, N - 5)
        for i in range(5):
            freq_intervals[0].append(nu[i] - delta)
            freq_intervals[1].append(nu[i] + delta)
    return freq_intervals

#################################
def Func(X, Y):
    return 1 + X - sp.exp(-X ** 2) + Y ** 2

def FindMean(x, y, U):
    N = len(x)
    mean = .0
    for i in range(N):
        U[i] = Func(x[i], y[i])
        mean += U[i]
    mean = mean / N
    return mean

def Graph(x, y):
    p1 = plt.plot(x, y, 'ro')
    plt.show()

def WritingInFile(names, sequences, fileName):
    with open(fileName, 'w') as f:
        for i in range(len(names)):
            f.write(names[i] + ':\n')
            for j in range(len(sequences[i])):
                f.write('\t' + str(sequences[i][j]) + '\n')

def FindResponds(x1, x2, outputFile, N):
    p = 0.08
    w2 = .0
    dispers = .0
    U = np.zeros(N)
    y = np.zeros(N)
    tr = .0

    mean = FindMean(x1, x2, U)
    
    for i in range(N):        
        tr = U[i] - mean
        w2 += tr ** 2

    w2 = w2 / (N - 1)
    dispers = math.sqrt(p * w2)
    ej = np.random.normal(0, dispers, N)

    for i in range(N):
        y[i] = U[i] + ej[i]
    
    WritingInFile(['U', 'ej', 'y'], [U, ej, y], outputFile)

    return y, dispers