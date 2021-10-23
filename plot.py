#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from os import path
from config import RESULTS_PATH, PHOTO_ERR_LOG_FILE


def create_graph(file, ylabel, name):
    with open(file) as f:
        data = f.readlines()

    errors = [float(x[:-1]) for x in data]

    plt.plot(errors)
    plt.xlabel("iterations")
    plt.ylabel(ylabel)
    plt.savefig(path.join(RESULTS_PATH, name))
    plt.close()


create_graph(PHOTO_ERR_LOG_FILE, "Photometric Error", "photo_error.png")
