import numpy as np
import os
import pandas as pd
import torch
from qpsolvers import Problem, solve_problem
import scipy
from sklearn.svm import SVC, OneClassSVM, LinearSVC, LinearSVR
from specification.NeuralEmbedding import NeuralEmbeddingSpecification

def solve_qp(K: np.ndarray, C: np.ndarray):
    """Solver for the following quadratic programming(QP) problem:
        - min   1/2 x^T K x - C^T x
        s.t     1^T x - 1 = 0
                - I x <= 0

    Parameters
    ----------
    K : np.ndarray
            Parameter in the quadratic term.
    C : np.ndarray
            Parameter in the linear term.

    Returns
    -------
    torch.tensor
            Solution to the QP problem.
    """
    if torch.is_tensor(K):
        K = K.cpu().numpy()
    if torch.is_tensor(C):
        C = C.cpu().numpy()
    C = C.reshape(-1)

    n = K.shape[0]
    P = np.array(K)
    P = scipy.sparse.csc_matrix(P)
    q = np.array(-C)
    G = np.array(-np.eye(n))
    G = scipy.sparse.csc_matrix(G)
    h = np.array(np.zeros((n, 1)))
    A = np.array(np.ones((1, n)))
    A = scipy.sparse.csc_matrix(A)
    b = np.array(np.ones((1, 1)))

    problem = Problem(P, q, G, h, A, b)
    solution = solve_problem(problem, solver="clarabel")
    w = solution.x
    w = torch.from_numpy(w).reshape(-1)
    return w, solution.obj


class toy_weight_estimation():
    def __init__(self, Phi: list):
        self.Phi = Phi

    def sim_eval(self, Z, X):
        Z = np.array(Z)
        X = np.array(X)
        v = np.zeros((Z.shape[0], X.shape[0]))
        for i in range(Z.shape[0]):
            z_i = Z[i].reshape(1, -1)
            v[i, :] = np.linalg.norm(z_i - X[:, np.newaxis], axis=2).reshape(1, -1)
        return v

    def specification_eval(self, Z: np.ndarray, X: np.ndarray):
        Z = np.array(Z)
        X = np.array(X)
        v = np.zeros((Z.shape[0], X.shape[0]))
        for i in range(Z.shape[0]):
            z_i = Z[i].reshape(1, -1)
            v[i, :] = z_i @ X.T
        return v

    def calculate_mean_or_cov(self, Z: np.ndarray, label: np.ndarray):
        Z = np.array(Z)
        label = np.array(label)
        df = pd.DataFrame(Z)
        df['label'] = label
        mean_by_label = df.groupby('label').mean()
        cov_matrix = df.groupby('label').cov()
        cov_matrix = np.array(cov_matrix)
        step_size = int(cov_matrix.shape[0] / len(np.unique(label)))

        cov_by_label = []

        for i in range(0, cov_matrix.shape[0], step_size):
            sub_cov = cov_matrix[i: i + step_size]
            cov_by_label.append(sub_cov)

        return np.array(mean_by_label), cov_by_label

    def weight_estiamtion(self, task_data: np.ndarray):
        C = []
        N = task_data.shape[0]
        self.specification_mean = []
        # self.specification_cov = []
        for i in range(len(self.Phi)):
            cal_mean, cal_cov = self.calculate_mean_or_cov(self.Phi[i].z.cpu().detach().numpy(), self.Phi[i].label.cpu().detach().numpy())
            self.specification_mean.append(cal_mean)
            v = self.specification_eval(cal_mean, task_data)
            C.append(np.sum(v, axis=1))

        C = np.concatenate(C, axis=0)
        C = C / N
        C = C.reshape(-1, 1)
        H = np.zeros((C.shape[0], C.shape[0]))
        class_all_specification = np.concatenate(self.specification_mean)
        count_i = 0
        for i, Phi in enumerate(self.specification_mean):
            for j in range(Phi.shape[0]):
                Z_i = np.array(Phi[j]).reshape(1, -1)
                Z_j = np.array(class_all_specification)
                K = Z_i @ Z_j.T
                H[count_i, :] = K
                count_i += 1

        w_all, _ = solve_qp(H, C)
        w_all = np.array(w_all).reshape(-1)

        weight_estimate = np.zeros(len(self.Phi))
        count_i = 0
        class_weight = []
        for i in range(len(self.Phi)):
            class_num = len(np.unique(self.Phi[i].label.cpu().detach().numpy()))
            count = 0
            record = np.zeros((1, class_num))
            for j in range(class_num):
                count += w_all[count_i]
                record[0, j] = w_all[count_i]
                count_i += 1
            class_weight.append(record)
            weight_estimate[i] = count

        return weight_estimate, class_weight


class Oracle():
    # The oracle binary classifier
    # Predicts 1 inside the circle, 0 outside the circle.
    def __init__(self, Radius):
        self.radius = Radius

    def predict(self, X):
        n = X.shape[0]
        y = np.zeros(n)
        for i in range(n):
            x = X[i, :]
            # positive inside circle
            if np.linalg.norm(x) <= self.radius:
                y[i] = 1
        return y


def generate_toy_data(Radius):
    h = Radius * np.sqrt(2) / 2
    center_list = [[Radius, 0], [h, h], [0, Radius], [-h, h], [-Radius, 0], [-h, -h], [0, -Radius], [h, -h]]
    X_list = []
    sigma = 0.2  # default: 0.2
    N = 80
    for center in center_list:
        mu = np.array(center)
        X_list.append(np.random.normal(loc=mu, scale=sigma, size=(N, 2)))

    X = [0, 0, 0, 0]
    X[0] = np.concatenate((X_list[0], X_list[1]), axis=0)
    X[1] = np.concatenate((X_list[2], X_list[3]), axis=0)
    X[2] = np.concatenate((X_list[4], X_list[5]), axis=0)
    X[3] = np.concatenate((X_list[6], X_list[7]), axis=0)

    test_N = 1000
    weight = [0.7, 0.7, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0]
    inst_list = []
    Yinst_list = []
    task_list = []
    for Y_num, (i, j) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):
        nb = int(round(test_N * weight[i] / 2))
        X_i = np.random.normal(loc=center_list[i], scale=sigma, size=(nb, 2))
        X_j = np.random.normal(loc=center_list[j], scale=sigma, size=(nb, 2))
        inst_list.append(np.concatenate((X_i, X_j), axis=0))
        Yinst_list.append([Y_num for _ in range(nb*2)])
        if i <= 4:
            task_list.append(np.ones(nb * 2) * int(i / 2))
        else:
            task_list.append(np.ones(nb * 2) * 10000)  # new class

    Xt_inst = np.concatenate(inst_list, axis=0)
    Yt_inst = np.concatenate(Yinst_list, axis=0)
    Task_list = np.concatenate(task_list, axis=0)

    test_n = int(round(test_N / 2))
    X1 = np.random.normal(loc=center_list[2], scale=sigma, size=(test_n, 2))
    X2 = np.random.normal(loc=center_list[3], scale=sigma, size=(test_n, 2))
    Xt_task = np.concatenate((X1, X2), axis=0)

    for i in range(len(X)):
        np.random.shuffle(X[i])
    np.random.shuffle(Xt_task)

    # shuffle_index = np.arange(Xt_inst.shape[0])
    # np.random.shuffle(shuffle_index)
    # Xt_inst = Xt_inst[shuffle_index, :]
    # Yt_inst = Yt_inst[shuffle_index, :]
    Task_inst = Task_list

    return X, Xt_task, Xt_inst, Task_inst, Yt_inst


def developer_submitting(Radius):
    # Generate toy data
    X, Xt_task, Xt_inst, Task_inst, Ytinst = generate_toy_data(Radius)
    X = [X[0], X[1], X[2]]
    c = len(X)
    oracle = Oracle(Radius)
    Y = [None, None, None, None]
    for i in range(c):
        Y[i] = oracle.predict(X[i])
    Yt_task = oracle.predict(Xt_task)
    Yt_inst = oracle.predict(Xt_inst)
    for i in range(Xt_inst.shape[0]):
        if Task_inst[i] == 10000:
            Yt_inst[i] = 10000

    # Pre-trained models
    f = []
    pseudo_Y = []
    for i in range(c):
        f_local = SVC(kernel='rbf', gamma=2, probability=True,
                      decision_function_shape='ovr')
        f_local.fit(X[i], Y[i])
        predict_Y = f_local.predict(X[i])
        f.append(f_local)
        pseudo_Y.append(predict_Y)

    return f, X, pseudo_Y, Y, Xt_task, Xt_inst, Ytinst


def run(Radius):
    """ The submitted stage"""
    """ Generating developer task and pre-trained model """
    f, X, pseudo_Y, Y, Xt_task, Xt_inst, Ytinst = developer_submitting(Radius)

    """ Specification generation and submitting to learnware market """
    K = 10
    Phi = []
    for i in range(len(X)):
        print(f'_____________task_num:{i}________________')
        spec = NeuralEmbeddingSpecification()
        spec.generate_state_spec_from_data(X=X[i], K=K, channel=X[i].shape[1], pseudo_labels=pseudo_Y[i], true_labels=Y[i])
        Phi.append(spec)

    """ The deployed stage """
    """ Mixture weight estimation """
    w_estimation = toy_weight_estimation(Phi)
    print('True weight: (0.7,0.3,0.0,0.0)')
    estimate_weight, class_weight = w_estimation.weight_estiamtion(Xt_inst)
    print('Estimated weight: (%.3f,%.3f,%.3f,%.3f)' % (
    estimate_weight[0], estimate_weight[1], estimate_weight[2], estimate_weight[3]))



if __name__ == '__main__':
    Radius = 1
    run(Radius)


