#! /usr/bin/env python
# -*- coding: utf-8 -*-

import fire
import pandas as pd
import numpy as np

from collections import defaultdict

from theoc.oc import load_result
from theoc.metrics import l2_error
from theoc.metrics import discrete_entropy
from theoc.metrics import discrete_mutual_information
from theoc.metrics import normalize

from theagamma.util import select_n
from theagamma.util import to_spikemat


def run(output_name, n, *file_names):

    # params
    dt = 1e-3  # ms resolution
    m = 8
    simultation_time = 6

    H = defaultdict(list)
    MI = defaultdict(list)
    dMI = defaultdict(list)
    l2 = defaultdict(list)
    dl2 = defaultdict(list)

    # Subsample files and do MI on each
    for i, file_name in enumerate(file_names):
        res = load_result(file_name)

        # Common ref
        y_ref = res["norm_rates"]["stim_ref"]

        # Stim calcs
        ts, ns = res["spikes"]["stim_p"]
        idx = np.random.randint(ns.min(), ns.max(), size=n)
        ns, ts = select_n(idx, ns, ts)
        # Convert to rates
        mat = to_spikemat(ns, ts, simultation_time, ns.max(), dt)
        # Convert to y, do MI
        y = normalize(mat.sum(1))
        MI["stim_p"].append(discrete_mutual_information(y_ref, y, m))
        H["stim_p"].append(discrete_entropy(y, m))
        l2["stim_p"].append(l2_error(y_ref, y))

        # E calcs
        ts, ns = res["spikes"]["E"]
        idx = np.random.randint(ns.min(), ns.max(), size=n)
        ns, ts = select_n(idx, ns, ts)
        # Convert to rates
        mat = to_spikemat(ns, ts, simultation_time, ns.max(), dt)
        # Convert to y, do MI
        y = normalize(mat.sum(1))
        MI["E"].append(discrete_mutual_information(y_ref, y, m))
        H["E"].append(discrete_entropy(y, m))
        l2["E"].append(l2_error(y_ref, y))

        # Delta MI
        dMI["stim_p"].append(MI["stim_p"][-1] - MI["stim_p"][-1])
        dMI["E"].append(MI["E"][-1] - MI["stim_p"][-1])
        dl2["E"].append(l2["E"][-1] - l2["stim_p"][-1])

        # Metadata
        H["file_index"].append(i)
        MI["file_index"].append(i)
        dMI["file_index"].append(i)
        l2["file_index"].append(i)
        dl2["file_index"].append(i)
        H["file_name"].append(file_name)
        MI["file_name"].append(file_name)
        dMI["file_name"].append(file_name)
        l2["file_name"].append(file_name)
        dl2["file_name"].append(file_name)
        H["num_pop"].append(n)
        MI["num_pop"].append(n)
        dMI["num_pop"].append(n)
        l2["num_pop"].append(n)
        dl2["num_pop"].append(n)

    # -- Dump to disk!
    df_H = pd.DataFrame(H)
    df_H.to_csv(output_name + f"_{n}_H.csv", index=False)
    df_MI = pd.DataFrame(MI)
    df_MI.to_csv(output_name + f"_{n}_MI.csv", index=False)
    df_dMI = pd.DataFrame(dMI)
    df_dMI.to_csv(output_name + f"_{n}_dMI.csv", index=False)
    df_l2 = pd.DataFrame(l2)
    df_l2.to_csv(output_name + f"_{n}_l2.csv", index=False)
    df_dl2 = pd.DataFrame(dl2)
    df_dl2.to_csv(output_name + f"_{n}_dl2.csv", index=False)


if __name__ == "__main__":
    # Create a command line interface automatically...
    fire.Fire(run)
