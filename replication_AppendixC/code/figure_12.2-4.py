import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import csv
import time
import itertools
from numpy import random
from operator import itemgetter
import importlib
import multiprocessing as mp
import json

import experiment_helper
import exp_wrapper

import finite_sim as sim
import finite_sim_wrapper as sim_wrapper

importlib.reload(exp_wrapper)
importlib.reload(experiment_helper)
importlib.reload(sim)
importlib.reload(sim_wrapper)

alpha = 1
epsilon = 1
tau = 1
lams = [.1,1,10]


tsr_ac_al_values = [.5]
cr_a_C = 0.5
lr_a_L = 0.5

tsr_est_types = ['tsr_est_naive', 'tsri_1.0','tsri_2.0']

customer_types = ['c1', 'c2']
listing_types = ['l']
exp_conditions = ['control', 'treat']
rhos_pre_treat = {'l':1} #adds up to 1
customer_proportions = {'c1':0.5,'c2':0.5} #adds up to 1

# used for multiplicative utilities
customer_type_base = {'c1':.12,'c2':.46}
listing_type_base = {'l':1}
exp_condition_base = {'control':1, 'treat':1.25}

vs = {}
for c in customer_types:
    vs[c] = {}
    vs[c]['treat'] = {}
    vs[c]['control'] = {}
    for l in listing_types:
        for e in exp_conditions:
            vs[c][e][l] = round(customer_type_base[c]
                                *exp_condition_base[e]
                                *listing_type_base[l],4)


tsr_bias_mf = {}
lr_bias_mf = {}
cr_bias_mf = {}
global_control_booking_probs_mf = {}
global_treat_booking_probs_mf = {}
gc_occupancy = {}
gtes = {}
gtes_norm = {}

global_solutions = {}
lr_solutions = {}
cr_solutions = {}
tsr_solutions = {}
tsr_fixed_solutions = {}

tsr_full_results = {}
tsr_fixed_full_results = {}

lr_params = {}
cr_params = {}
tsr_params = {}
tsr_fixed_params = {}

a_Cs = {}
a_Ls = {}
betas = {}
scales = {}
cr_side_weights = {}

tsr_fixed_customer_comp = {}
tsr_fixed_listing_comp = {}

norm_constants = {}
# normalize estimates and booking rates by min(lam, tau) to control for size of market
for lam in lams:
    norm_constants[lam] = 1 / min(lam, tau)

for lam in lams:
    betas[lam] =  np.exp(-1*lam/tau)
    cr_side_weights[lam] = np.exp(-lam/tau)
    a_Cs[lam] = 1/2 * betas[lam] + 1*(1-betas[lam])
    a_Ls[lam] = 1 * betas[lam] + 1/2 * (1-betas[lam])
    scales[lam] = betas[lam]*(1-betas[lam])







T_0 = 5
T_1 = 25

# normalizes time horizon by min(lam, tau)
T_start = {lam: T_0/min(lam,tau) for lam in lams}
T_end = {lam: T_1/min(lam,tau) for lam in lams}

varying_time_horizons = True

n_runs = 70
n_listings = 700

choice_set_type = 'alpha'
k = None






if __name__ == '__main__':
    params = sim_wrapper.calc_all_params(listing_types, rhos_pre_treat,
                                         customer_types, customer_proportions, vs,
                                         alpha, epsilon, tau, lams,
                                         tsr_ac_al_values, cr_a_C, lr_a_L)
    events = sim_wrapper.run_all_sims(n_runs, n_listings, T_start, T_end,
                                      choice_set_type, k,
                                      alpha, epsilon, tau, lams,
                                      **params)
    est_stats = sim_wrapper.calc_all_ests_stats("sample", T_start, T_end,
                                                n_listings, tau=tau,
                                                tsr_est_types=tsr_est_types,
                                                events=events,
                                                varying_time_horizons=varying_time_horizons,
                                                **params
                                                )
    est_stats = est_stats.rename({'cr': "Customer-Side", 'lr': "Listing-Side",
                                  'tsri_2.0': 'TSRI-2', 'tsri_1.0': 'TSRI-1',
                                  'tsr_est_naive': "TSR-Naive"
                                  })

    sim_bias = ((est_stats
          .unstack(level=1)['abs_bias_over_GTE'])
          .abs()
          ).T

    sim_se = ((est_stats
          .unstack(level=1)['std_over_GTE'])
          .abs()
          ).T

    sim_rmse = ((est_stats
          .unstack(level=1)['rmse_over_GTE'])
          .abs()
          ).T



    rgb_dict = {'cr': (0, 114, 178), 'lr': (255, 127, 14),
                'tsrn': (0, 158, 115), 'tsri1': (240, 228, 66),
                'tsri2': (204, 121, 167),
                'cluster': (86, 180, 233)}
    rgb_0_1_array = np.array(list(rgb_dict.values())) / 255
    ax = plt.figure().add_subplot(111)

    # Bias plot
    fig, ax = plt.subplots()
    sim_bias.plot(ax=ax, kind='bar', color=rgb_0_1_array)
    plt.xlabel("Relative Demand $\lambda/\\tau$")
    plt.yscale('log')
    plt.legend(loc=[1.02, 0.5])
    plt.title("Simulations: Bias")
    plt.ylabel("Abs (Bias/GTE)")
    plt.tight_layout()
    sns.despine()
    plt.savefig('12.2.png')
    plt.show()

    # SE plot
    fig, ax = plt.subplots()
    sim_se.plot(ax=ax, kind='bar', color=rgb_0_1_array)
    plt.xlabel("Relative Demand $\lambda/\\tau$")
    plt.legend(loc=[1.02, 0.5])
    plt.title("Simulations: Standard Error")
    plt.ylabel("Standard Error/GTE")
    plt.tight_layout()
    sns.despine()
    plt.savefig('12.3.png')
    plt.show()

    # RMSE plot
    fig, ax = plt.subplots()
    sim_rmse.plot(ax=ax, kind='bar', color=rgb_0_1_array)
    plt.xlabel("Relative Demand $\lambda/\\tau$")
    plt.yscale('log')
    plt.legend(loc=[1.02, 0.5])
    plt.title("Simulations: RMSE")
    plt.ylabel("RMSE/GTE")
    plt.tight_layout()
    sns.despine()
    plt.savefig('12.4.png')
    plt.show()
















































