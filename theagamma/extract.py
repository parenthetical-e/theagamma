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


def subsample(output_name, n, *file_names):

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


def run(output_name, *file_names):
    """Run several OC experiments, saving select results to disk"""

    # Init
    H = defaultdict(list)
    MI = defaultdict(list)
    dMI = defaultdict(list)
    power = defaultdict(list)
    center = defaultdict(list)

    # -- Run

    for i, file_name in enumerate(file_names):
        print(file_name)
        res = load_result(file_name)

        # Save parts of the result...
        #
        # Entopy
        for b in res['H'].keys():
            H[b].append(res["H"][b])
        H["trial"].append(i)

        # MI
        for b in res['MI'].keys():
            MI[b].append(res['MI'][b])
        MI["trial"].append(i)

        # Change in MI, dMI
        for b in res['dMI'].keys():
            dMI[b].append(res['dMI'][b])
        dMI["trial"].append(i)

        # Peak power
        for b in res['power'].keys():
            power[b].append(res['power'][b])
        power["trial"].append(i)

        # Peak center
        for b in res['center'].keys():
            center[b].append(res['center'][b])
        center["trial"].append(i)

    # -- Dump to disk!
    df_H = pd.DataFrame(H)
    df_H.to_csv(output_name + "_H.csv", index=False)
    df_MI = pd.DataFrame(MI)
    df_MI.to_csv(output_name + "_MI.csv", index=False)
    df_dMI = pd.DataFrame(dMI)
    df_dMI.to_csv(output_name + "_dMI.csv", index=False)
    df_power = pd.DataFrame(power)
    df_power.to_csv(output_name + "_power.csv", index=False)
    df_center = pd.DataFrame(center)
    df_center.to_csv(output_name + "_center.csv", index=False)

    return None


if __name__ == "__main__":
    # Create a command line interface automatically...
    fire.Fire(run)
