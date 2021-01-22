#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : bayesian_optimization.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 11:06 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 11:06 下午 by shendu.ht  init
"""
from collections import OrderedDict

import numpy
from joblib import Parallel, delayed
from scipy.optimize import minimize


class BayesianOptimization:

    def __init__(self, gp, acq, func, param_dict, n_job=1):
        """
        Bayesian Optimization class.

        Args:
            gp: model instance
                Gaussian process model instance
            acquisition: Acquisition instance
                Acquisition instance.
            func: function
                Function to maximize over parameters specified by `param_dict`
            param_dict: dict
                Dictionary specifying parameter, type and bounds.
            n_job: int
                Default 1. Parallel threads to use during acquisition optimization.
        """
        self.eps = 10e-6
        self.gp = gp
        self.acq = acq
        self.func = func
        self.params = param_dict
        self.n_jobs = n_job

        self.params_key = list(param_dict.keys())
        self.params_value = list(param_dict.values())
        self.params_type = [p[0] for p in self.params_value]
        self.params_range = [p[1] for p in self.params_value]

        self.history = []

    def _sample_param(self):
        """
        Randomly sample parameters over bounds

        Returns:
            dict:
                A random sample of specified parameters.
        """

        d = OrderedDict()
        for index, param in enumerate(self.params_key):
            if self.params_type[index] == 'int':
                d[param] = numpy.random.randint(
                    self.params_range[index][0], self.params_range[index][1])
            elif self.params_type[index] == 'cont':
                d[param] = numpy.random.uniform(
                    self.params_range[index][0], self.params_range[index][1])
            else:
                raise ValueError('Unsupported variable type.')
        return d

    def _first_run(self, n_eval=3):
        """
        Performs initial evaluations before fitting gaussian process.

        Args:
            n_eval: int
                Number of initial evaluations to perform. Default is 3.
        """
        self.x = numpy.empty((n_eval, len(self.params_key)))
        self.y = numpy.empty((n_eval,))
        for i in range(n_eval):
            s_param = self._sample_param()
            s_param_val = list(s_param.values())
            self.x[i] = s_param_val
            self.y[i] = self.func(**s_param)
        self.gp.fit(self.x, self.y)
        self.tau = numpy.max(self.y)
        self.history.append(self.tau)

    def _acq_wrapper(self, x_new):
        """
        Evaluates the acquisition function on a point.

        Args:
            x_new: np.ndarray, shape=((len(self.params_key),))
                Point to evaluate the acquisition function on.

        Returns:
            float
                Acquisition function value for `x_new`
        """
        new_mean, new_var = self.gp.predict(x_new, return_std=True)
        new_std = numpy.sqrt(new_var + self.eps)
        return -self.acq.eval(self.tau, new_mean, new_std)

    def _optimize_acq(self, method="L-BFGS-B", n_start=100):
        """
        Optimizes the acquisition function using a multistart approach.

        Args:
            method: str
                Any `scipy.optimize` method that admits bounds and gradients. Default is 'L-BFGS-B'.
            n_start: int
                Number of starting points for the optimization procedure. Default is 100.
        """
        start_points_dict = [self._sample_param() for _ in range(n_start)]
        start_points_arr = numpy.array([list(s.values()) for s in start_points_dict])
        x_best = numpy.empty((n_start, len(self.params_key)))
        f_best = numpy.empty((n_start,))
        if self.n_jobs == 1:
            for index, start_point in enumerate(start_points_arr):
                res = minimize(self._acq_wrapper, x0=start_point, method=method,
                               bounds=self.params_range)
                x_best[index], f_best[index] = res.x, numpy.atleast_1d(res.fun)[0]
        else:
            opt = Parallel(n_jobs=self.n_jobs)(delayed(minimize)(self._acq_wrapper,
                                                                 x0=start_point,
                                                                 method=method,
                                                                 bounds=self.params_range) for start_point in
                                               start_points_arr)
            x_best = numpy.array([res.x for res in opt])
            f_best = numpy.array([numpy.atleast_1d(res.fun)[0] for res in opt])

        self.best = x_best[numpy.argmin(f_best)]

    def update_gp(self):
        """
        Updates the internal model with the next acquired point and its evaluation.
        """
        kw = {param: self.best[i] for i, param in enumerate(self.params_key)}
        f_new = self.func(**kw)
        self.gp.update(numpy.atleast_2d(self.best), numpy.atleast_1d(f_new))
        self.tau = numpy.max(self.gp.y)
        self.history.append(self.tau)

    def get_result(self):
        """
        Prints best result in the Bayesian Optimization procedure.

        Returns:
            OrderedDict
                Point yielding best evaluation in the procedure.
            float
                Best function evaluation.
        """
        arg_tau = numpy.argmax(self.gp.y)
        opt_x = self.gp.x[arg_tau]
        res_d = OrderedDict()
        for i, (key, param_type) in enumerate(zip(self.params_key, self.params_type)):
            if param_type == 'int':
                res_d[key] = int(opt_x[i])
            else:
                res_d[key] = opt_x[i]
        return res_d, self.tau

    def run(self, max_iter=10, init_eval=3, resume=False):
        """
        Runs the Bayesian Optimization procedure.

        Args:
            max_iter: int
                Number of iterations to run. Default is 10.
            init_eval: int
                Initial function evaluations before fitting a GP. Default is 3.
            resume: bool
                Whether to resume the optimization procedure from the last evaluation. Default is `False`.
        """
        if not resume:
            self.init_eval = init_eval
            self._first_run(self.init_eval)
        for iteration in range(max_iter):
            self._optimize_acq()
            self.update_gp()
