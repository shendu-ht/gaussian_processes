#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : acquisition_test.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 12:11 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 12:11 下午 by shendu.ht  init
"""
import unittest

import numpy

from bayes_opt.acquisition import Acquisition


class TestAcquisition(unittest.TestCase):
    """
    Basic test for acquisition
    """

    def testMode(self):
        """
        test acq mode
        """
        tau = 1.96
        mean = 0
        std = 1

        modes = ['ExpectedImprovement', 'ProbabilityImprovement', 'UCB', 'Entropy', 'tExpectedImprovement']

        for mode in modes:
            acq = Acquisition(mode=mode)
            print(mode, acq.eval(tau, mean, std))

    def testMcmcMode(self):
        """
        test acq mode for mcmc
        """
        tau = 1.96
        means = numpy.random.randn(1000)
        stds = numpy.random.uniform(0.8, 1.2, 1000)
        modes_mcmc = ['IntegratedExpectedImprovement', 'IntegratedProbabilityImprovement',
                      'IntegratedUCB', 'tIntegratedExpectedImprovement']

        for mode in modes_mcmc:
            acq = Acquisition(mode=mode)
            print(mode, acq.eval(tau, means, stds))


if __name__ == '__main__':
    unittest.main()
