#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : cov_func.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 5:23 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 5:23 下午 by shendu.ht  init
"""
from abc import ABC, abstractmethod

import numpy
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma

default_bounds = {
    'l': [1e-4, 1],
    'sigma_f': [1e-6, 2],
    'sigma_n': [1e-6, 2],
    'v': [1e-3, 10],
    'gamma': [1e-3, 1.99],
    'alpha': [1e-3, 1e4],
    'period': [1e-3, 10]
}


def l2norm_(x, x_star):
    """
    Wrapper function to compute the L2 norm

    Args:
        x: numpy.ndarray, shape=(n, n_features)
            Instances.
        x_star: numpy.ndarray, shape=(m, n_features)
            Instances.

    Returns:
        numpy.ndarray
            Pairwise euclidean distance between row pairs of `x` and `x_star`.
    """
    return cdist(x, x_star)


def kron_delta(x, x_star):
    """
    Computes Kronecker delta for rows in x and x_star.

    Args:
        x: numpy.ndarray, shape=(n, n_features)
            Instances.
        x_star: numpy.ndarray, shape=(m, n_features)
            Instances.

    Returns:
        numpy.ndarray
            Kronecker delta between row pairs of `x` and `x_star`.
    """
    return cdist(x, x_star) < numpy.finfo(numpy.float32).eps


class Kernel(ABC):
    """
    Abstract Base Class for gaussian process cov functions
    """

    @abstractmethod
    def __init__(self, v=1, l=1.0, sigma_f=1.0, sigma_n=1e-6, bounds=None, parameters=None):
        """
        Args:
            v: float
                Scale-mixture hyperparameter of the Matern covariance function.
            l: float
                Characteristic length-scale. Units in input space in which posterior GP values do not
                change significantly.
            sigma_f: float
                Signal variance. Controls the overall scale of the covariance function.
            sigma_n: float
                Noise variance. Additive noise in output space.
            bounds: list
                List of tuples specifying hyperparameter range in optimization procedure.
            parameters: list
                List of strings specifying which hyperparameters should be optimized.
        """
        self.v = v
        self.l = l
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n

        if parameters is None:
            parameters = ['l', 'sigma_f', 'sigma_n']
        self.parameters = parameters

        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    @abstractmethod
    def k(self, x, x_star):
        """
        Computes covariance function values over `x` and `x_star`.

        Args:
            x: numpy.ndarray, shape=(n, n_features)
                Instances.
            x_star: numpy.ndarray, shape=(m, n_features)
                Instances.

        Returns:
            numpy.ndarray
                Computed covariance matrix.
        """
        pass

    @abstractmethod
    def grad_k(self, x, x_star, param):
        """
        Computes gradient matrix for instances `x`, `x_star` and hyperparameter `param`.

        Args:
            x: numpy.ndarray, shape=(n, n_features)
                Instances.
            x_star: numpy.ndarray, shape=(m, n_features)
                Instances.
            param: str
                Parameter to compute gradient matrix for.

        Returns:
            np.ndarray
                Gradient matrix for parameter `param`.
        """
        pass


class SquaredExponential(Kernel):
    """
    Squared exponential kernel class.
    """

    def __init__(self, l=1, sigma_f=1.0, sigma_n=1e-6, bounds=None, parameters=None):
        if parameters is None:
            parameters = ['l', 'sigma_f', 'sigma_n']
        super().__init__(l=l, sigma_f=sigma_f, sigma_n=sigma_n, bounds=bounds, parameters=parameters)

    def k(self, x, x_star):
        r = l2norm_(x, x_star)
        return self.sigma_f * numpy.exp(-.5 * r ** 2 / self.l ** 2) + self.sigma_n * kron_delta(x, x_star)

    def grad_k(self, x, x_star, param='l'):
        if param == 'l':
            r = l2norm_(x, x_star)
            num = r ** 2 * self.sigma_f * numpy.exp(-r ** 2 / (2 * self.l ** 2))
            den = self.l ** 3
            l_grad = num / den
            return l_grad

        elif param == 'sigma_f':
            r = l2norm_(x, x_star)
            sigma_f_grad = (numpy.exp(-.5 * r ** 2 / self.l ** 2))
            return sigma_f_grad

        elif param == 'sigma_n':
            sigma_n_grad = kron_delta(x, x_star)
            return sigma_n_grad

        else:
            raise ValueError('Param not found')


class Matern(Kernel):
    """
    Matern kernel class.
    """

    def __init__(self, v=1, l=1, sigma_f=1, sigma_n=1e-6, bounds=None, parameters=None):
        if parameters is None:
            parameters = ['v', 'l', 'sigma_f', 'sigma_n']
        super().__init__(v=v, l=l, sigma_f=sigma_f, sigma_n=sigma_n, bounds=bounds, parameters=parameters)

    def k(self, x, x_star):
        r = l2norm_(x, x_star)
        bessel = kv(self.v, numpy.sqrt(2 * self.v) * r / self.l)
        f = 2 ** (1 - self.v) / gamma(self.v) * (numpy.sqrt(2 * self.v) * r / self.l) ** self.v
        res = f * bessel
        res[numpy.isnan(res)] = 1
        res = self.sigma_f * res + self.sigma_n * kron_delta(x, x_star)
        return res

    def grad_k(self, x, x_star, param):
        pass


class Matern32(Kernel):
    """
    Matern v=3/2 kernel class.
    """

    def __init__(self, l=1, sigma_f=1, sigma_n=1e-6, bounds=None, parameters=None):
        if parameters is None:
            parameters = ['l', 'sigma_f', 'sigma_n']
        super().__init__(l=l, sigma_f=sigma_f, sigma_n=sigma_n, bounds=bounds, parameters=parameters)

    def k(self, x, x_star):
        r = l2norm_(x, x_star)
        one = (1 + numpy.sqrt(3 * (r / self.l) ** 2))
        two = numpy.exp(- numpy.sqrt(3 * (r / self.l) ** 2))
        return self.sigma_f * one * two + self.sigma_n * kron_delta(x, x_star)

    def grad_k(self, x, x_star, param):
        if param == 'l':
            r = l2norm_(x, x_star)
            num = 3 * (r ** 2) * self.sigma_f * numpy.exp(-numpy.sqrt(3) * r / self.l)
            return num / (self.l ** 3)
        elif param == 'sigma_f':
            r = l2norm_(x, x_star)
            one = (1 + numpy.sqrt(3 * (r / self.l) ** 2))
            two = numpy.exp(- numpy.sqrt(3 * (r / self.l) ** 2))
            return one * two
        elif param == 'sigma_n':
            return kron_delta(x, x_star)
        else:
            raise ValueError('Param not found')


class Matern52(Kernel):
    """
    Matern v=5/2 kernel class.
    """

    def __init__(self, l=1, sigma_f=1, sigma_n=1e-6, bounds=None, parameters=None):
        if parameters is None:
            parameters = ['l', 'sigma_f', 'sigma_n']
        super().__init__(l=l, sigma_f=sigma_f, sigma_n=sigma_n, bounds=bounds, parameters=parameters)

    def k(self, x, x_star):
        r = l2norm_(x, x_star) / self.l
        one = (1 + numpy.sqrt(5 * r ** 2) + 5 * r ** 2 / 3)
        two = numpy.exp(-numpy.sqrt(5 * r ** 2))
        return self.sigma_f * one * two + self.sigma_n * kron_delta(x, x_star)

    def grad_k(self, x, x_star, param):
        r = l2norm_(x, x_star)
        if param == 'l':
            num_one = 5 * r ** 2 * numpy.exp(-numpy.sqrt(5) * r / self.l)
            num_two = numpy.sqrt(5) * r / self.l + 1
            res = num_one * num_two / (3 * self.l ** 3)
            return res
        elif param == 'sigma_f':
            one = (1 + numpy.sqrt(5 * (r / self.l) ** 2) + 5 * (r / self.l) ** 2 / 3)
            two = numpy.exp(-numpy.sqrt(5 * r ** 2))
            return one * two
        elif param == 'sigma_n':
            return kron_delta(x, x_star)
        else:
            raise ValueError('Param not found')


class GammaExponential(Kernel):
    """
    Gamma-exponential kernel class.
    """

    def __init__(self, gamma=1, l=1, sigma_f=1, sigma_n=1e-6, bounds=None, parameters=None):
        """

        Args:
            gamma: float
                Hyperparameter of the Gamma-exponential covariance function.
        """
        if parameters is None:
            parameters = ['gamma', 'l', 'sigma_f', 'sigma_n']
        super().__init__(l=l, sigma_f=sigma_f, sigma_n=sigma_n, bounds=bounds, parameters=parameters)
        self.gamma = gamma

    def k(self, x, x_star):
        r = l2norm_(x, x_star)
        return self.sigma_f * (numpy.exp(-(r / self.l) ** self.gamma)) + self.sigma_n * kron_delta(x, x_star)

    def grad_k(self, x, x_star, param):
        if param == 'gamma':
            eps = 10e-6
            r = l2norm_(x, x_star) + eps
            first = -numpy.exp(- (r / self.l) ** self.gamma)
            sec = (r / self.l) ** self.gamma * numpy.log(r / self.l)
            gamma_grad = first * sec
            return gamma_grad
        elif param == 'l':
            r = l2norm_(x, x_star)
            num = self.gamma * numpy.exp(-(r / self.l) ** self.gamma) * (r / self.l) ** self.gamma
            l_grad = num / self.l
            return l_grad
        elif param == 'sigma_f':
            r = l2norm_(x, x_star)
            sigma_f_grad = (numpy.exp(-(r / self.l) ** self.gamma))
            return sigma_f_grad
        elif param == 'sigma_n':
            sigma_n_grad = kron_delta(x, x_star)
            return sigma_n_grad
        else:
            raise ValueError('Param not found')


class RationalQuadratic(Kernel):
    """
    Rational-quadratic kernel class.
    """

    def __init__(self, alpha=1, l=1, sigma_f=1, sigma_n=1e-6, bounds=None, parameters=None):
        """

        Args:
            alpha: float
                Hyperparameter of the rational-quadratic covariance function.
        """
        if parameters is None:
            parameters = ['alpha', 'l', 'sigma_f', 'sigma_n']
        super().__init__(l=l, sigma_f=sigma_f, sigma_n=sigma_n, bounds=bounds, parameters=parameters)
        self.alpha = alpha

    def k(self, x, x_star):
        r = l2norm_(x, x_star)
        return self.sigma_f * ((1 + r ** 2 / (2 * self.alpha * self.l ** 2)) **
                               (-self.alpha)) + self.sigma_n * kron_delta(x, x_star)

    def grad_k(self, x, x_star, param):
        if param == 'alpha':
            r = l2norm_(x, x_star)
            one = (r ** 2 / (2 * self.alpha * self.l ** 2) + 1) ** (-self.alpha)
            two = r ** 2 / ((2 * self.alpha * self.l ** 2) * (r ** 2 / (2 * self.alpha * self.l ** 2) + 1))
            three = numpy.log(r ** 2 / (2 * self.alpha * self.l ** 2) + 1)
            alpha_grad = one * (two - three)
            return alpha_grad
        elif param == 'l':
            r = l2norm_(x, x_star)
            num = r ** 2 * (r ** 2 / (2 * self.alpha * self.l ** 2) + 1) ** (-self.alpha - 1)
            l_grad = num / self.l ** 3
            return l_grad
        elif param == 'sigma_f':
            r = l2norm_(x, x_star)
            sigma_f_grad = (1 + r ** 2 / (2 * self.alpha * self.l ** 2)) ** (-self.alpha)
            return sigma_f_grad
        elif param == 'sigma_n':
            sigma_n_grad = kron_delta(x, x_star)
            return sigma_n_grad
        else:
            raise ValueError('Param not found')


class ExpSine(Kernel):
    """
    Exponential sine kernel class.
    """

    def __init__(self, l=1.0, period=1.0, bounds=None, parameters=None):
        if parameters is None:
            parameters = ['l', 'period']
        super().__init__(l=l, bounds=bounds, parameters=parameters)
        self.period = period

    def k(self, x, x_star):
        r = l2norm_(x, x_star)
        num = - 2 * numpy.sin(numpy.pi * r / self.period)
        return numpy.exp(num / self.l) ** 2 + 1e-4

    def grad_k(self, x, x_star, param):
        if param == 'l':
            r = l2norm_(x, x_star)
            one = 4 * numpy.sin(numpy.pi * r / self.period)
            two = numpy.exp(-4 * numpy.sin(numpy.pi * r / self.period) / self.l)
            return one * two / (self.l ** 2)
        elif param == 'period':
            r = l2norm_(x, x_star)
            one = 4 * numpy.pi * r * numpy.cos(numpy.pi * r / self.period)
            two = numpy.exp(-4 * numpy.sin(numpy.pi * r / self.period) / self.l)
            return one * two / (self.l * self.period ** 2)
        else:
            raise ValueError('Param not found')


class DotProd(Kernel):
    def __init__(self, sigma_f=1.0, sigma_n=1e-6, bounds=None, parameters=None):
        if parameters is None:
            parameters = ['sigma_f', 'sigma_n']
        super().__init__(sigma_f=sigma_f, sigma_n=sigma_n, bounds=bounds, parameters=parameters)

    def k(self, x, x_star):
        return self.sigma_f * numpy.dot(x, x_star.T) + self.sigma_n * kron_delta(x, x_star)

    def grad_k(self, x, x_star, param):
        if param == 'sigma_f':
            return numpy.dot(x, x_star.T)
        elif param == 'sigma_n':
            return self.sigma_f * numpy.dot(x, x_star.T)
