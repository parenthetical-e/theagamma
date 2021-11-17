#! /usr/bin/env python
# -*- coding: utf-8 -*-

import fire
import sys
import pandas as pd
import os
import numpy as np

from joblib import Parallel, delayed
from itertools import product
from collections import defaultdict

from theoc.oc import load_result
from theoc.metrics import discrete_dist
from theoc.metrics import discrete_entropy
from theoc.metrics import discrete_mutual_information
from theoc.metrics import normalize

from theagamma.util import select_n
from theagamma.util import bin_times
from theagamma.util import to_spikemat


def run(output_name, n, *file_names):

    # params
    dt = 1e-3  # ms resolution
    m = 8

    H = defaultdict(list)
    MI = defaultdict(list)
    dMI = defaultdict(list)

    # Subsample files and do MI on each
    for i, file_name in enumerate(file_names):
        res = load_result(file_name)

        # Common ref
        y_ref = res["norm_rates"]["stim_ref"]

        # Stim calcss
        y = res["norm_rates"]["stim_p"]
        MI["stim_p"].append(discrete_mutual_information(y_ref, y, m))
        H["stim_p"].append(discrete_entropy(y, m))

        # E calcs
        ts, ns = res["spikes"]["E"]
        idx = np.random.random_integers(ns.min(), ns.max(), size=n)
        ns, ts = select_n(idx, ns, ts)
        # Convert to rates
        mat = to_spikemat(ns, ts, ts.max(), ns.max(), dt)
        # Convert to y, do MI
        y = normalize(mat.sum(1))
        MI["E"].append(discrete_mutual_information(y_ref, y, m))
        H["E"].append(discrete_entropy(y, m))

        # Delta MI
        dMI["stim_p"].append(MI["stim_p"][-1] - MI["stim_p"][-1])
        dMI["E"].append(MI["E"][-1] - MI["stim_p"][-1])

        # Metadata
        H["trial"].append(i)
        MI["trial"].append(i)
        dMI["trial"].append(i)
        H["file_name"].append(file_name)
        MI["file_name"].append(file_name)
        dMI["file_name"].append(file_name)

    # -- Dump to disk!
    df_H = pd.DataFrame(H)
    df_H.to_csv(output_name + f"_{n}_H.csv", index=False)
    df_MI = pd.DataFrame(MI)
    df_MI.to_csv(output_name + f"{n}_MI.csv", index=False)
    df_dMI = pd.DataFrame(dMI)
    df_dMI.to_csv(output_name + f"{n}_dMI.csv", index=False)


if __name__ == "__main__":
    # Create a command line interface automatically...
    fire.Fire(run)
