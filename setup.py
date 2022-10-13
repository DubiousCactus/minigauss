#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from setuptools import find_packages, setup

setup(
    name="minigauss",
    packages=find_packages(include=["minigauss", "minigauss.priors"]),
    version="0.1.0",
    description="Mini Gaussian Process library",
    author="Theo Morales",
    license="MIT",
    install_requires=["tqdm", "numpy"],
)
