#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 16:22:22 2025

@author: rzty
"""

import numpy as np
import matplotlib.pyplot as plt

# Load the heading error data
heading_file_paths = [
    "heading_errors_limit_25_1.npy",
    "heading_errors_limit_25_2.npy",
    "heading_errors_limit_25_3.npy"
]

position_file_paths = [
    'position_errors_limit_25_1.npy',
    'position_errors_limit_25_2.npy',
    'position_errors_limit_25_3.npy'
    ]

# Load and stack all error arrays
error_arrays = [np.load(path) for path in heading_file_paths]
errors_stacked = np.stack(error_arrays)

error_arrays_2 = [np.load(path) for path in position_file_paths]
errors_stacked_2 = np.stack(error_arrays_2)

# Compute the average error across the runs
average_error = np.mean(errors_stacked, axis=0)
average_error_2 = np.mean(errors_stacked_2, axis=0)

# Plotting the average error
fig, axs = plt.subplots(1,2,figsize = (14,5))

test_name = 'limits_25'

title = 'Average Heading Error in '+test_name
title_2 = 'Average Position Error in '+test_name
axs[0].plot(average_error, label=title)
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Heading Error (radians)')
axs[0].set_title(title+' over Time')
axs[0].legend()
axs[0].grid(True)


axs[1].plot(average_error_2, label=title_2)
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Position Error (meters)')
axs[1].set_title(title_2+' Over Time')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
