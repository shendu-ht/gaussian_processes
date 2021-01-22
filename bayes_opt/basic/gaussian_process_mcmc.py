#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : gaussian_process_mcmc.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 4:33 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 4:33 下午 by shendu.ht  init
"""
import numpy
import pymc3
from theano import tensor

from bayes_opt.basic.gaussian_process import GaussianProcess

cov_equivalence = {'SquaredExponential': pymc3.gp.cov.ExpQuad,
                   'Matern52': pymc3.gp.cov.Matern52,
                   'Matern32': pymc3.gp.cov.Matern32}


class GaussianProcessMcmc:

    def __init__(self, cov_func, n_iter=2000, burn_in=1000, init='ADVI', step=None):
        """
        Gaussian Process class using MCMC sampling of covariance function hyperparameters.

        Args:
            cov_func:
                Covariance function to use. Currently this instance only supports `squaredExponential`
                `matern32` and `matern52` kernels.
            n_iter: int
                Number of iterations to run MCMC.
            burn_in: int
                Burn-in iterations to discard at trace.
            init: str
                Initialization method for NUTS.
            step:
                pyMC3's step method for the process, (e.g. `pm.Slice`)
        """

        self.cov_func = cov_func
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.init = init
        self.step = step

    def _extract_param(self, unit_trace, cov_params):
        d = {}
        for key, value in unit_trace.items():
            if key in cov_params:
                d[key] = value
        if 'v' in cov_params:
            d['v'] = 5 / 2
        return d

    def fit(self, x, y):
        """
        Fits a Gaussian Process regressor using MCMC.

        Args:
            x: np.ndarray, shape=(n_samples, n_features)
            Training instances to fit the Gaussian process.
        y: np.ndarray, shape=(n_samples,)
            Corresponding continuous target values to `x`.
        """
        self.x = x
        self.n = self.x.shape[0]
        self.y = y
        self.model = pymc3.Model()

        with self.model as model:
            l = pymc3.Uniform('l', 0, 10)

            log_s2_f = pymc3.Uniform('log_s2_f', lower=-7, upper=5)
            s2_f = pymc3.Deterministic('sigmaf', tensor.exp(log_s2_f))

            log_s2_n = pymc3.Uniform('log_s2_n', lower=-7, upper=5)
            s2_n = pymc3.Deterministic('sigman', tensor.exp(log_s2_n))

            f_cov = s2_f * cov_equivalence[type(self.cov_func).__name__](1, l)
            sigma = f_cov(self.x) + tensor.eye(self.n) * s2_n ** 2
            y_obs = pymc3.MvNormal('y_obs', mu=numpy.zeros(self.n), cov=sigma, observed=self.y)
        with self.model as model:
            if self.step is not None:
                self.trace = pymc3.sample(self.n_iter, step=self.step())[self.burn_in:]
            else:
                self.trace = pymc3.sample(self.n_iter, init=self.init)[self.burn_in:]

    def predict(self, x_star, return_std=False, n_samples=10):
        """
        Returns mean and covariances for each posterior sampled Gaussian Process.

        Args:
            x_star: np.ndarray, shape=((n_samples, n_features))
                Testing instances to predict.
            return_std: bool
                Whether to return the standard deviation of the posterior process. Otherwise,
                it returns the whole covariance matrix of the posterior process.
            n_samples: int
                Number of posterior MCMC samples to consider.

        Returns:
            np.ndarray
                Mean of the posterior process for each MCMC sample and `x_star`.
            np.ndarray
                Covariance posterior process for each MCMC sample and `x_star`.
        """
        chunk = list(self.trace)
        chunk = chunk[::-1][:n_samples]
        post_mean = []
        post_var = []
        for posterior_sample in chunk:
            params = self._extract_param(posterior_sample, self.cov_func.parameters)
            cov_func = self.cov_func.__class__(**params)
            gp = GaussianProcess(cov_func)
            gp.fit(self.x, self.y)
            m, s = gp.predict(x_star, return_std=return_std)
            post_mean.append(m)
            post_var.append(s)
        return numpy.array(post_mean), numpy.array(post_var)

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
