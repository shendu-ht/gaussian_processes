#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : acquisition.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 4:30 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 4:30 下午 by shendu.ht  init
"""
import numpy
from scipy.stats import norm, t


class Acquisition:

    def __init__(self, mode, eps=1e-6, **params):
        """
        Acquisition function class.

        Args:
            mode: str
                Define the behaviour of the acquisition strategy. Currently support:
                    `ExpectedImprovement`, `IntegratedExpectedÌmprovement`, `ProbabilityImprovement`
                    `IntegratedProbabilityImprovement`, `UCB`, `IntegratedUCB`, `Entropy`
                    `tExpectedImprovement`, and `tIntegratedExpectedImprovement`
            eps: float
                Small floating value to avoid `np.sqrt` or zero-division warnings.
            params: dict
                Extra parameters needed for certain acquisition functions,
                e.g. UCB needs to be supplied with `beta`.
        """
        self.params = params
        self.eps = eps

        mode_dict = {"ExpectedImprovement": self.expected_improvement,
                     "IntegratedExpectedImprovement": self.integrated_expected_improvement,
                     "ProbabilityImprovement": self.probability_improvement,
                     "IntegratedProbabilityImprovement": self.integrated_probability_improvement,
                     "UCB": self.ucb,
                     "IntegratedUCB": self.integrated_ucb,
                     "Entropy": self.entropy,
                     "tExpectedImprovement": self.t_expected_improvement,
                     "tIntegratedExpectedImprovement": self.t_integrated_expected_improvement}
        self.function = mode_dict[mode]

    def expected_improvement(self, tau, mean, std):
        """
        Expected Improvement acquisition function.

        Args:
            tau: float
                Best observed function evaluation.
            mean: float
                Point mean of the posterior process.
            std: float
                Point std of the posterior process.

        Returns:
            float
                Expected improvement.
        """
        z = (mean - tau - self.eps) / (std + self.eps)
        return (mean - tau) * norm.cdf(z) + std * norm.pdf(z)

    def integrated_expected_improvement(self, tau, mean_mcmc, std_mcmc):
        """
        Integrated expected improvement. Can only be used with `gaussian_process_mcmc`.

        Args:
            tau: float
                Best observed function evaluation
            mean_mcmc: array-like
                Means of posterior predictive distributions after sampling.
            std_mcmc: array-like
                Standard deviations of posterior predictive distributions after sampling.

        Returns:
            float:
                Integrated Expected Improvement
        """
        acq = [self.expected_improvement(tau, mean, std) for mean, std in zip(mean_mcmc, std_mcmc)]
        return numpy.average(acq)

    def probability_improvement(self, tau, mean, std):
        """
        Probability of Improvement acquisition function.

        Args:
            tau: float
                Best observed function evaluation.
            mean: float
                Point mean of the posterior process.
            std: float
                Point std of the posterior process.

        Returns:
            float
                Probability of improvement.
        """
        z = (mean - tau - self.eps) / (std + self.eps)
        return norm.cdf(z)

    def integrated_probability_improvement(self, tau, mean_mcmc, std_mcmc):
        """
        Integrated probability of improvement. Can only be used with `gaussian_process_mc`.

        Args:
            tau: float
                Best observed function evaluation
            mean_mcmc: array-like
                Means of posterior predictive distributions after sampling.
            std_mcmc: array-like
                Standard deviations of posterior predictive distributions after sampling.

        Returns:
            float:
                Integrated Probability of Improvement
        """
        acq = [self.probability_improvement(tau, mean, std) for mean, std in zip(mean_mcmc, std_mcmc)]
        return numpy.average(acq)

    def ucb(self, tau, mean, std, beta=1.5):
        """
        Upper-confidence bound acquisition function.

        Args:
            tau: float
                Best observed function evaluation.
            mean: float
                Point mean of the posterior process.
            std: float
                Point std of the posterior process.
            beta: float
                Hyperparameter controlling exploitation/exploration ratio.

        Returns:
            float
                Upper confidence bound.
        """
        return mean + beta * std

    def integrated_ucb(self, tau, mean_mcmc, std_mcmc, beta=1.5):
        """
        Integrated upper-confidence bound acquisition function. Can only be used with `gaussian_process_mc`.

        Args:
            tau: float
                Best observed function evaluation
            mean_mcmc: array-like
                Means of posterior predictive distributions after sampling.
            std_mcmc: array-like
                Standard deviations of posterior predictive distributions after sampling.
            beta: float
                Hyperparameter controlling exploitation/exploration ratio.

        Returns:
            float:
                Integrated UCB.
        """
        acq = [self.ucb(tau, mean, std, beta) for mean, std in zip(mean_mcmc, std_mcmc)]
        return numpy.average(acq)

    def entropy(self, tau, mean, std, sigma_n=1.0):
        """
        Predictive entropy acquisition function.

        Args:
            tau: float
                Best observed function evaluation.
            mean: float
                Point mean of the posterior process.
            std: float
                Point std of the posterior process.
            sigma_n: float
                Noise variance

        Returns:
            float:
                Predictive entropy.
        """
        sp2 = std ** 2 + sigma_n
        return 0.5 * numpy.log(2 * numpy.pi * numpy.e * sp2)

    def t_expected_improvement(self, tau, mean, std, nu=3.0):
        """
        Expected Improvement acquisition function. Only to be used with `t_student_process`.

        Args:
            tau: float
                Best observed function evaluation.
            mean: float
                Point mean of the posterior process.
            std: float
                Point std of the posterior process.
            nu: float
                Degrees of freedom.

        Returns:
            float
                Expected improvement.
        """
        gamma = (mean - tau - self.eps) / (std + self.eps)
        return gamma * std * t.cdf(gamma, df=nu) + std * (1 + (gamma ** 2 - 1) / (nu - 1)) * t.pdf(gamma, df=nu)

    def t_integrated_expected_improvement(self, tau, mean_mcmc, std_mcmc, nu=3.0):
        """
        Integrated expected improvement. Only to be used with `t_student_process_mcmc`.

        Args:
            tau: float
                Best observed function evaluation
            mean_mcmc: array-like
                Means of posterior predictive distributions after sampling.
            std_mcmc: array-like
                Standard deviations of posterior predictive distributions after sampling.
            nu:

        Returns:
            float
                Degrees of freedom.
        """
        acq = [self.t_expected_improvement(tau, mean, std, nu=nu) for mean, std in zip(mean_mcmc, std_mcmc)]
        return numpy.average(acq)

    def eval(self, tau, mean, std):
        """
        Evaluates selected acquisition function.

        Args:
            tau: float
                Best observed function evaluation.
            mean: (float, array-like)
                Point mean of the posterior process.
            std: (float, array-like)
                Point std of the posterior process.

        Returns:
            float
                Acquisition function value.
        """
        return self.function(tau, mean, std, **self.params)
