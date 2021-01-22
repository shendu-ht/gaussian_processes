#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : gaussian_process.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 4:45 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 4:45 下午 by shendu.ht  init
"""
from collections import OrderedDict

import numpy
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize


class GaussianProcess:

    def __init__(self, cov_func, optimize=False, use_grads=False, mean_prior=0):
        """
        Gaussian Process regressor class. Based on Rasmussen & Williams[1]

        Args:
            cov_func: object
                Internal covariance function.
            optimize: bool
                User chosen optimization configuration.
            use_grads: bool
                Gradient behavior.
            mean_prior: float
                Explicit value for the mean function of the prior Gaussian Process.

        Notes:
            [1] Rasmussen, C. E., & Williams, C. K. I. (2004). Gaussian processes for machine learning.
            International journal of neural systems (Vol. 14). http://doi.org/10.1142/S0129065704001899
        """

        self.cov_func = cov_func
        self.optimize = optimize
        self.use_grads = use_grads
        self.mean_prior = mean_prior

    def get_cov_params(self):
        """
        Current covariance function hyperparameters

        Returns:
            d: dict
                Dictionary containing covariance function hyperparameters
        """
        d = {}
        for param in self.cov_func.parameters:
            d[param] = self.cov_func.__dict__[param]
        return d

    def fit(self, x, y):
        """
        Fits a Gaussian Process regressor

        Args:
            x: numpy.ndarray, shape=(n_samples, n_features)
                Training instances to fit the Gaussian Process
            y: numpy.ndarray, shape=(n_samples, )
                Corresponding continuous target values to x
        """

        self.x = x
        self.y = y
        self.n_samples = self.x.shape[0]

        if self.optimize:
            grads = None
            if self.use_grads:
                grads = self._grads
            self.opt_hyp(param_key=self.cov_func.parameters, param_bounds=self.cov_func.bounds, grads=grads)

        self.k = self.cov_func.k(self.x, self.x)
        self.l = cholesky(self.k).T
        self.alpha = solve(self.l.T, solve(self.l, y - self.mean_prior))
        self.log_p = -0.5 * numpy.dot(self.y, self.alpha) - numpy.sum(
            numpy.log(numpy.diag(self.l))) - self.n_samples / 2 * numpy.log(2 * numpy.pi)

    def param_grad(self, k_param):
        """
        It is recommended to use `self._grad` instead.

        Args:
            k_param: dict
                Dictionary with keys being hyperparameters and values their queried values.

        Returns:
            np.ndarray
                Gradient corresponding to each hyperparameters. Order given by `k_param.keys()`
        """
        k_param_key = list(k_param.keys())
        cov_func = self.cov_func.__class__(**k_param)
        k = cov_func.k(self.x, self.x)
        l = cholesky(k).T
        alpha = solve(l.T, solve(l, self.y))
        inner = numpy.dot(numpy.atleast_2d(alpha).T, numpy.atleast_2d(alpha)) - numpy.linalg.inv(k)
        grads = []
        for param in k_param_key:
            grad_k = cov_func.grad_k(self.x, self.x, param=param)
            grad_k = .5 * numpy.trace(numpy.dot(inner, grad_k))
            grads.append(grad_k)
        return numpy.array(grads)

    def _lmlik(self, param_vector, param_key):
        """
        Calculate marginal negative log-likelihood for given covariance hyperparameters.

        Args:
            param_vector: list
                List of values corresponding to hyperparameters to query.
            param_key: list
                List of hyperparameter strings corresponding to `param_vector`.

        Returns:
            float
                Negative log-marginal likelihood for chosen hyperparameters.
        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        self.cov_func = self.cov_func.__class__(**k_param)

        # This fixes recursion
        original_opt = self.optimize
        original_grad = self.use_grads
        self.optimize = False
        self.use_grads = False

        self.fit(self.x, self.y)

        self.optimize = original_opt
        self.use_grads = original_grad
        return - self.log_p

    def _grads(self, param_vector, param_key):
        """
        Calculate gradient for each hyperparameter, evaluated at a given point.

        Args:
            param_vector: list
                List of values corresponding to hyperparameters to query.
            param_key: list
                List of hyperparameter strings corresponding to `param_vector`.

        Returns:
            np.ndarray
                Gradient for each evaluated hyperparameter.
        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        return - self.param_grad(k_param)

    def opt_hyp(self, param_key, param_bounds, grads=None, n_trials=5):
        """
        Optimizes the negative marginal log-likelihood for given hyperparameters and bounds.
        This is an empirical Bayes approach (or Type II maximum-likelihood).

        Args:
            param_key: list
                List of hyperparameters to optimize.
            param_bounds: list
                List containing tuples defining bounds for each hyperparameter to optimize over.
        """
        xs = [[1, 1, 1]]
        fs = [self._lmlik(xs[0], param_key)]
        for trial in range(n_trials):
            x0 = []
            for param, bound in zip(param_key, param_bounds):
                x0.append(numpy.random.uniform(bound[0], bound[1], 1)[0])
            if grads is None:
                res = minimize(self._lmlik, x0=x0, args=param_key, method='L-BFGS-B', bounds=param_bounds)
            else:
                res = minimize(self._lmlik, x0=x0, args=param_key, method='L-BFGS-B', bounds=param_bounds, jac=grads)
            xs.append(res.x)
            fs.append(res.fun)

        arg_min = numpy.argmin(fs)
        opt_param = xs[arg_min]
        k_param = OrderedDict()
        for k, x in zip(param_key, opt_param):
            k_param[k] = x
        self.cov_func = self.cov_func.__class__(**k_param)

    def predict(self, x_star, return_std=False):
        """
        Returns mean and covariances for the posterior Gaussian Process.

        Args:
            x_star: np.ndarray, shape=((n_samples, n_features))
                Testing instances to predict.
            return_std: bool
                Whether to return the standard deviation of the posterior process. Otherwise,
                it returns the whole covariance matrix of the posterior process.

        Returns:
            np.ndarray
                Mean of the posterior process for testing instances.
            np.ndarray
                Covariance of the posterior process for testing instances.
        """
        x_star = numpy.atleast_2d(x_star)
        k_star = self.cov_func.k(self.x, x_star).T
        f_mean = self.mean_prior + numpy.dot(k_star, self.alpha)
        v = solve(self.l, k_star.T)
        f_cov = self.cov_func.k(x_star, x_star) - numpy.dot(v.T, v)
        if return_std:
            f_cov = numpy.diag(f_cov)
        return f_mean, f_cov

    def update(self, x_new, y_new):
        """
        Updates the internal model with `x_new` and `y_new` instances.

        Args:
            x_new: np.ndarray, shape=((m, n_features))
                New training instances to update the model with.
            y_new: np.ndarray, shape=((m,))
                New training targets to update the model with.
        """
        y = numpy.concatenate((self.y, y_new), axis=0)
        x = numpy.concatenate((self.x, x_new), axis=0)
        self.fit(x, y)
