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
            **params: dict
                Extra parameters needed for certain acquisition functions,
                e.g. UCB needs to be supplied with `beta`.
        """
