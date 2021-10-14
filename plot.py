#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from config import PERR_LOG_FILE, RESULTS_PATH

with open(PERR_LOG_FILE) as f:
    data = f.readlines()

errors = [float(x[:-1]) for x in data]

plt.plot(errors)
plt.savefig(RESULTS_PATH+"graph.png")
