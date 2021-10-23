#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from os import path
import sys
from config import LOGS_PATH

if len(sys.argv) > 1:
    folder = sys.argv[1]
    LOGS_PATH = path.join(folder, "logs_dfvs")
    PHOTO_ERR_LOG_FILE = path.join(LOGS_PATH, "p_err.txt")
    PHOTO_ERR_PLOT = path.join(LOGS_PATH, "p_err.png")


def create_graph(file, ylabel, name):
    with open(file) as f:
        data = f.readlines()

    errors = [float(x[:-1]) for x in data]

    plt.plot(errors)
    plt.xlabel("iterations")
    plt.ylabel(ylabel)
    plt.savefig(name)
    plt.close()


create_graph(PHOTO_ERR_LOG_FILE, "Photometric Error", PHOTO_ERR_PLOT)
os.system(f"ln -sf {PHOTO_ERR_PLOT} p_err.png")
