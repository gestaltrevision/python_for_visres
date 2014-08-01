#! /usr/bin/env python
from psychopy_ext import ui

__author__ = "Jonas Kubilius"
__version__ = "0.1"
exp_choices = [
    ui.Choices('scripts.trivial', name='Quick demo'),
    ui.Choices('scripts.changedet', name='Change Detection Experiment')
    ]

# bring up the graphic user interface or interpret command line inputs
# usually you can skip the size parameter
ui.Control(exp_choices, title='Demo Project', size=(560,550))
