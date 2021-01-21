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
import numpy
from scipy.linalg import cholesky, solve


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

    def _grads(self, param_vector, param_key):
        """

        Args:
            param_vector:
            param_key:

        Returns:

        """

    def opt_hyp(self, param_key, param_bounds, grads=None, n_trials=5):
        """

        Args:
            param_key:
            param_bounds:
            grads:
            n_trials:

        Returns:

        """
