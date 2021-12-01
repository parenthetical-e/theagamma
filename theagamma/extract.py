#! /usr/bin/env python
# -*- coding: utf-8 -*-

import fire
import pandas as pd
from collections import defaultdict
from theoc.oc import load_result


def run(output_name, *file_names):
    """Run several OC experiments, saving select results to disk"""

    # Init
    H = defaultdict(list)
    MI = defaultdict(list)
    dMI = defaultdict(list)
    l2 = defaultdict(list)
    dl2 = defaultdict(list)
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

        # l2 error
        for b in res['l2'].keys():
            l2[b].append(res['l2'][b])
        l2["trial"].append(i)

        # Change in MI, dMI
        for b in res['dl2'].keys():
            dl2[b].append(res['dl2'][b])
        dl2["trial"].append(i)

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
    df_l2 = pd.DataFrame(l2)
    df_l2.to_csv(output_name + "_l2.csv", index=False)
    df_dl2 = pd.DataFrame(dl2)
    df_dl2.to_csv(output_name + "_dl2.csv", index=False)
    df_power = pd.DataFrame(power)
    df_power.to_csv(output_name + "_power.csv", index=False)
    df_center = pd.DataFrame(center)
    df_center.to_csv(output_name + "_center.csv", index=False)

    return None


if __name__ == "__main__":
    # Create a command line interface automatically...
    fire.Fire(run)
