#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : cov_func_test.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 3:50 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 3:50 下午 by shendu.ht  init
"""
import unittest

import numpy

from bayes_opt.basic.cov_func import SquaredExponential, Matern, Matern32, Matern52, GammaExponential, \
    RationalQuadratic, ExpSine, DotProd

cov_funcs = [SquaredExponential(), Matern(), Matern32(), Matern52(), GammaExponential(),
             RationalQuadratic(), ExpSine(), DotProd()]

grad_enabled = [SquaredExponential(), Matern32(), Matern52(), GammaExponential(), RationalQuadratic(),
                ExpSine(), DotProd()]

cov_classes = dict(SquaredExponential=SquaredExponential, Matern=Matern, Matern32=Matern32,
                   Matern52=Matern52, GammaExponential=GammaExponential,
                   RationalQuadratic=RationalQuadratic, DotProd=DotProd)

hyper_param_interval = dict(SquaredExponential=dict(l=(0, 2.0), sigma_f=(0, 0.5), sigma_n=(0, 0.5)),
                            Matern=dict(l=(0, 2.0), sigma_f=(0, 0.5), sigma_n=(0, 0.5)),
                            Matern32=dict(l=(0, 2.0), sigma_f=(0, 0.5), sigma_n=(0, 0.5)),
                            Matern52=dict(l=(0, 2.0), sigma_f=(0, 0.5), sigma_n=(0, 0.5)),
                            GammaExponential=dict(gamma=(0, 2.0), l=(0, 2.0), sigma_f=(0, 0.5),
                                                  sigma_n=(0, 0.5)),
                            RationalQuadratic=dict(alpha=(0, 2.0), l=(0, 2.0), sigma_f=(0, 0.5),
                                                   sigma_n=(0, 0.5)),
                            DotProd=dict(sigma_f=(0, 0.5), sigma_n=(0, 0.5)))


def generate_hyper_param(**hyper_param_interval):
    hyper_params = dict()

    for hyper_param, bound in hyper_param_interval.items():
        hyper_params[hyper_param] = numpy.random.uniform(bound[0], bound[1])

    return hyper_params


class TestCovFunc(unittest.TestCase):
    """
    Basic test for cov func
    """

    def testSim(self):
        """
        test covariance function values computing
        """
        x = numpy.random.randn(100, 3)
        for cov in cov_funcs:
            cov.k(x, x)

    def testGrad(self):
        """
        test gradient matrix computing
        """
        x = numpy.random.randn(3, 3)
        for cov in grad_enabled:
            for param in cov.parameters:
                cov.grad_k(x, x, param=param)

    def testPsdCovFunc(self):
        """
        test if generated covariance functions are positive definite
        """
        for name in cov_classes:
            for i in range(10):
                hyper_params = generate_hyper_param(**hyper_param_interval[name])
                cov = cov_classes[name](**hyper_params)
                for j in range(1000):
                    x = numpy.random.randn(10, 2)
                    eig = numpy.linalg.eigvals(cov.k(x, x))
                    assert (eig > 0).all()


if __name__ == '__main__':
    unittest.main()
