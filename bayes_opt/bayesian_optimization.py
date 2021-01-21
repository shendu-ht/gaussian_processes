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
import numpy
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn import gaussian_process


def expected_improvement(x, gp, eval_loss, is_maximise=False, n_params=1):
    """
    Expected improvement acquisition function.

    Args:
        x: array-like, shape = [n_samples, n_hyperparams].
            The point for which the expected improvement needs to be computed.
        gp: GaussianProcessRegressor.
            Gaussian process trained on previously evaluated hyperparameters.
        eval_loss: numpy.ndarray.
            the value of the target function for the previously evaluated hyperparameters.
        is_maximise: bool.
            Whether the target function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gp.predict(x_to_predict, return_std=True)

    if is_maximise:
        loss_opt = numpy.max(eval_loss)
        scaling_factor = 1
    else:
        loss_opt = numpy.min(eval_loss)
        scaling_factor = -1

    # In case sigma equals zero
    with numpy.errstate(divide="ignore"):
        z = scaling_factor * (mu - loss_opt) / sigma
        exp_improve = scaling_factor * (mu - loss_opt) * norm.cdf(z) + sigma * norm.pdf(z)
        exp_improve[sigma == 0] = 0
    return -1 * exp_improve


def sample_next_hyperparameter(acq_func, gp, eval_loss, is_maximise=False, bounds=(0, 10), n_restarts=25):
    """
    Proposes the next hyperparameter to sample the loss function for.

    Args:
        acq_func: function.
            Acquisition function to optimise.
        gp: GaussianProcessRegressor.
            Gaussian process trained on previously evaluated hyperparameters.
        eval_loss: numpy.ndarray.
            the value of the target function for the previously evaluated hyperparameters.
        is_maximise: bool.
            Whether the target function is to be maximised or minimised.
        bounds: tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: int.
            Number of times to run the minimiser with different starting points.
    """

    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for start_point in numpy.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):
        res = minimize(fun=acq_func, x0=start_point.reshape(1, -1), bounds=bounds, method='L-BFGS-B',
                       args=(gp, eval_loss, is_maximise, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x
    return best_x


def bayesian_optimization(n_iter, function, bounds, init_samples=None, n_pre_samples=5, gp_param=None,
                          random_search=False, alpha=1e-5, epsilon=1e-7):
    """
    Uses Gaussian Processes to optimise the loss function `sample_loss`

    Args:
        n_iter: int.
            Number of iterations to run the search algorithm.
        function: function.
            Target function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function.
        init_samples: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the function for. If None,
            randomly sample from the target function.
        n_pre_samples: int.
            If x is None, samples `n_pre_samples` initial points from the target function.
        gp_param: dict.
            Dict of parameters to pass on to the underlying Gaussian Process.
        random_search: bool.
            Whether to perform random search or L-BFGS-B optimisation over the acquisition function.
        alpha: double.
            Variance of the error term of the Gaussian Process.
        epsilon: double.
            Precision tolerance for floats.
    """

    x = []
    y = []

    n_params = bounds.shape[0]

    # get initial points
    if init_samples is None:
        for random_sample in numpy.random.uniform(bounds[:, 0], bounds[:1], (n_pre_samples, n_params)):
            x.append(random_sample)
            y.append(function(random_sample))
    else:
        for init_sample in init_samples:
            x.append(init_sample)
            y.append(function(init_sample))

    xp = numpy.array(x)
    yp = numpy.array(y)

    # create Gaussian Process
    if gp_param is None:
        kernel = gaussian_process.kernels.Matern()
        model = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10,
                                                          normalize_y=True)
    else:
        model = gaussian_process.GaussianProcessRegressor(**gp_param)

    # n_iter round iteration
    for n in range(n_iter):
        model.fit(xp, yp)

        # sample next hyperparameter
        if random_search:
            x_random = numpy.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, is_maximise=True, n_params=n_params)
            next_sample = x_random[numpy.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, is_maximise=True, bounds=bounds,
                                                     n_restarts=100)

        # Duplicates will break the gaussian process. If duplicate, randomly sample a next query point.
        if numpy.any(numpy.abs(next_sample - xp) <= epsilon):
            next_sample = numpy.random.uniform(bounds[:,0], bounds[:,1], n_params)

        # sample loss
        score = function(next_sample)

        # Update lists
        x.append(next_sample)
        y.append(score)

        xp = numpy.array(x)
        yp = numpy.array(y)
    return xp, yp
