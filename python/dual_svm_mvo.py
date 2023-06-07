import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from python.math_utils import *
from python.svm import *
from sklearn.model_selection import StratifiedShuffleSplit
from python.analysis_tools import plotSVC

pth = ''


# print("Input WLS Access ID")
# accessid = input()
# print("Input WLS License ID")
# licenseid = input()
# print("Input WLS Secret Key")
# secret_key = input()
# #web license try to access it via uoft
e = gp.Env(empty=True)
# e.setParam('OutputFlag', 0)
# e.setParam('WLSACCESSID', accessid)
# e.setParam('LICENSEID', int(licenseid))
# e.setParam('WLSSECRET', secret_key)
e.start()


class DualSVM:
    # This class models the support vector machine sub problem in the ADM method
    big_m = 100
    # noinspection PyTypeChecker
    def __init__(self, exogenous, soft_margin, previous_exogenous = None,
                 previous_alpha = None, previous_z = None, z=None, kernel = None, epsilon=1):

        self.exogenous = exogenous  # n by p matrix of features for the tickers
        self.previous_exogenous = previous_exogenous #data for previous period
        self.previous_alpha = previous_alpha #alpha for previous period
        self.previous_z = previous_z #previous labels
        self.soft_margin = soft_margin  # hyper parameter
        n, m = self.exogenous.shape

        self.model = gp.Model(env=e)
        self.kernel = kernel
        self.alpha = self.model.addMVar(n, lb=0, ub = self.soft_margin)
        self.z = z
        self.epsilon = epsilon

    @property
    def svm_objective(self):
        """
        dual svm objective
        $\boldsymbol{\alpha}^{\intercal} \boldsymbol{1}
        - (1/2) \boldsymbol{\alpha} Q \boldsymbol{\alpha}
        where Q_{ij} = u_i u_j Kernel(y_i, y_j)
        :return:
        """
        self.kernel.fit(self.exogenous)
        self.Q = self.kernel.transform(self.exogenous)
        self.u = 2*self.z - 1
        if len(self.u.shape) == 1:
            self.u = np.expand_dims(self.u, axis=1)
        U = self.u@self.u.T
        QU = np.multiply(self.Q, U)
        QU = nearestPD(QU)
        return self.epsilon*self.alpha.sum() - (1 / 2) * (self.alpha @ QU @ self.alpha)

    @property
    def svm_change(self):
        """
        this models an objective function term that captures the case
        where we are trying to limit how much our hyperplane changes
        :return:
        """
        n, m = self.exogenous.shape
        Q_ = self.kernel.transform(self.previous_exogenous)

        self.u_previous = 2*self.previous_z - 1
        if len(self.u_previous.shape) == 1:
            self.u_previous = np.expand_dims(self.u_previous, axis=1)
        if len(self.previous_alpha.shape) == 1:
            self.previous_alpha = np.expand_dims(self.previous_alpha, axis=1)
        if len(self.u.shape) == 1:
            self.u = np.expand_dims(self.u, axis=1)

        U = self.u_previous @ self.u.T
        QU = np.multiply(Q_, U)

        return (-1)*self.previous_alpha.T @ QU @ self.alpha

    def set_model(self, svm_constrs=None, delta=0):
        """
        sets the gurobi model
        takes in optional constraints
        :param svm_constrs:
        :param delta:
        :return:
        """
        self.model.remove(self.model.getConstrs())
        # parameter definitions

        # objective function components
        if svm_constrs:
            for con in svm_constrs:
                self.model.addConstr(con, 'target')

        hyperplane_penalty = self.svm_objective
        if delta != 0:
            hyperplane_penalty = self.svm_objective + self.svm_change

        self.model.setObjective(hyperplane_penalty, GRB.MAXIMIZE)

        self.model.addConstr(self.u.T@self.alpha == 0)
        #update bounds here

    def optimize(self, cbb=None):
        n, m = self.exogenous.shape
        if cbb not in [None]:
            self.model._cur_obj = float('inf')
            self.model._time = time.time()
            self.model.optimize(callback=cbb)
        else:
            self.model.optimize()

        if self.previous_alpha is None:
            v = np.multiply(np.expand_dims(self.alpha.x, axis = 1), self.u)
            for i in range(n):
                if self.alpha[i].x < self.soft_margin:
                    dot_product = self.Q[i,:]@v
                    self.b = self.u[i][0] - dot_product

            self.decision_boundary = self.Q@v + self.b
        else:
            v = np.multiply(np.expand_dims(self.alpha.x, axis = 1), self.u)
            Q_ = self.kernel.transform(self.previous_exogenous)
            print(Q_.shape)
            self.u_previous = 2*self.previous_z - 1
            if len(self.u_previous.shape) == 1:
                self.u_previous = np.expand_dims(self.u_previous, axis=1)

            if len(self.previous_alpha.shape) == 1:
                self.previous_alpha = np.expand_dims(self.previous_alpha, axis = 1)

            v_previous = np.multiply(self.previous_alpha, self.u_previous)
            for i in range(n):
                if self.alpha[i].x < self.soft_margin:
                    v = np.multiply(np.expand_dims(self.alpha.x, axis = 1), self.u)
                    dot_product = self.Q[i,:]@v
                    dot_product_previous = Q_[:,i].T@v_previous
                    self.b = self.u[i][0] - dot_product - dot_product_previous
            self.decision_boundary = self.Q@v + Q_.T@v_previous + self.b

    def predict(self, X):
        self.kernel.fit(self.exogenous)
        Q = self.kernel.transform(X)
        if self.previous_alpha is None:
            v = np.multiply(np.expand_dims(self.alpha.x, axis = 1), self.u)
            return np.sign(Q@v + self.b)
        else:
            v = np.multiply(np.expand_dims(self.alpha.x, axis = 1), self.u)
            self.kernel.fit(self.previous_exogenous)
            Q_  = self.kernel.transform(X)

            self.u_previous = 2*self.previous_z - 1

            if len(self.u_previous.shape) == 1:
                self.u_previous = np.expand_dims(self.u_previous, axis=1)
            if len(self.previous_alpha.shape) == 1:
                    self.previous_alpha = np.expand_dims(self.previous_alpha, axis = 1)

            v_previous = np.multiply(self.previous_alpha, self.u_previous)

            return np.sign(Q@v + Q_@v_previous + self.b)



def compute_best_classifer(X, y, plot = False):

    cv = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=42)
    C_range = np.logspace(-1, 5, 10, base = 5)
    gamma_range = np.logspace(-4, 1, 10, base=5)

    grid = train_svm(X, y, gamma_range, C_range, cv, plot=True)
    if plot:
        print("The best parameters are %s with a score of %0.2f"
            % (grid.best_params_, grid.best_score_))
    gamma = grid.best_params_['sbf__gamma']
    C = grid.best_params_['svm__C']
    kernel = CustomRBFKernel(gamma = gamma)

    SVM_ = DualSVM(X ,C, kernel =kernel)
    SVM_.z = y
    SVM_.set_model()
    SVM_.optimize()

    if plot:
        plotSVC('Testing', X, y, SVM_)

    return SVM_.alpha.x, gamma