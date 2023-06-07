from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt


def rbf_gram(X, Y, gamma):
    dists_sq = get_pairwise_distances(X, Y)
    # turn into an RBF gram matrix
    km = dists_sq;
    del dists_sq
    km *= -1 * gamma
    np.exp(km, km)  # exponentiates in-place
    return km


def get_pairwise_distances(X, Y):
    """

    :param X:
    :return:
    """
    # get a matrix where the (i, j)th element is |x[i] - x[j]|^2
    # using the identity (x - y)^T (x - y) = x^T x + y^T y - 2 x^T y
    n, p = X.shape
    pt_sq_norms_x = (X ** 2).sum(axis=1)
    pt_sq_norms_y = (Y ** 2).sum(axis=1)
    dists_sq = np.dot(X, Y.T)  # n by m
    dists_sq *= -2
    dists_sq += pt_sq_norms_x.reshape(-1, 1)
    dists_sq += pt_sq_norms_y
    # dists_sq = dists_sq/(p**2)
    return dists_sq


# Wrapper class for the custom kernel chi2_kernel
class CustomRBFKernel(BaseEstimator, TransformerMixin):
    def __init__(self, gamma=1.0):
        super(CustomRBFKernel, self).__init__()
        self.gamma = gamma

    def transform(self, X):
        return rbf_gram(X, self.X_train_, gamma=self.gamma)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


def train_svm(X, y, gamma_range, C_range, cv, plot=False):
    # Create a pipeline where our custom predefined kernel
    # is run before SVC.
    pipe = Pipeline([
        ('sbf', CustomRBFKernel()),
        ('svm', SVC()),
    ])

    # Set the parameter 'gamma' of our custom kernel by
    # using the 'estimator__param' syntax.

    cv_params = dict([
        ('sbf__gamma', gamma_range),
        ('svm__kernel', ['precomputed']),
        ('svm__C', C_range),
    ])

    # Do grid search to get the best parameter value of 'gamma'.
    grid = GridSearchCV(pipe, cv_params, cv=cv, scoring='accuracy')
    grid.fit(X, y)

    if plot:
        ax = plt.gca()
        DecisionBoundaryDisplay.from_estimator(
            grid,
            X,
            cmap=plt.cm.Paired,
            ax=ax,
            response_method="predict",
            plot_method="pcolormesh",
            shading="auto",
        )

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")
        plt.title("2-Class classification using Support Vector Machine with custom kernel")
        plt.axis("tight")
        plt.show()

        return grid