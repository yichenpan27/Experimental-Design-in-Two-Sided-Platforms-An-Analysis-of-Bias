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
lams = [1]


tsr_ac_al_values = [.5]
cr_a_C = 0.5
lr_a_L = 0.5

tsr_est_types = ['tsr_est_naive', 'tsri_1.0','tsri_2.0']

customer_types = ['c1']
listing_types = ['l']
exp_conditions = ['control', 'treat']
rhos_pre_treat = {'l':1} #adds up to 1
customer_proportions = {'c1':1} #adds up to 1

# used for multiplicative utilities
customer_type_base = {'c1':.155 }
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
for lam in lams:
    norm_constants[lam] = 1 / min(lam, tau)

for lam in lams:
    betas[lam] =  np.exp(-1*lam/tau)
    cr_side_weights[lam] = np.exp(-lam/tau)
    a_Cs[lam] = 1/2 * betas[lam] + 1*(1-betas[lam])
    a_Ls[lam] = 1 * betas[lam] + 1/2 * (1-betas[lam])
    scales[lam] = betas[lam]*(1-betas[lam])

for lam in lams:
    tsr_param = exp_wrapper.tsr_exp_induced_parameters(listing_types, rhos_pre_treat,
                                                       a_Cs[lam], a_Ls[lam],
                                                       customer_types, customer_proportions,
                                                       alpha, vs, lam)
    global_solution = exp_wrapper.calc_gte_from_exp_type("tsr", tau, epsilon,
                                                         **tsr_param)
    global_solutions[lam] = global_solution
    gtes[lam] = global_solution['gte']
    gtes_norm[lam] = global_solution['gte'] * norm_constants[lam]

    tsr_results = exp_wrapper.run_experiment("tsr", epsilon, tau, **tsr_param,
                                             a_C=a_Cs[lam], a_L=a_Ls[lam],
                                             customer_side_weight=cr_side_weights[lam],
                                             scale=scales[lam]
                                             )

    tsr_bias_mf[lam] = {}
    for tsr_est in tsr_est_types:
        tsr_bias_mf[lam][tsr_est] = ((tsr_results['estimator'][tsr_est] - gtes[lam])
                                     * norm_constants[lam])

    lr_exp_param = exp_wrapper.listing_side_exp_induced_parameters(listing_types, rhos_pre_treat, lr_a_L,
                                                                   customer_types,
                                                                   alpha, vs, lam, customer_proportions)
    lr_sol = exp_wrapper.run_experiment("listing", epsilon, tau, **lr_exp_param, a_L=lr_a_L)
    lr_bias_mf[lam] = abs(lr_sol['estimator']['naive_est'] - gtes[lam]) * norm_constants[lam]

    cr_exp_param = exp_wrapper.customer_side_exp_induced_parameters(listing_types, rhos_pre_treat, cr_a_C,
                                                                    customer_types, alpha, vs, lam,
                                                                    customer_proportions)
    cr_sol = exp_wrapper.run_experiment("customer", epsilon, tau, **cr_exp_param, a_C=cr_a_C)
    cr_bias_mf[lam] = abs(cr_sol['estimator']['naive_est'] - gtes[lam]) * norm_constants[lam]



mf_bias_df = pd.DataFrame([cr_bias_mf, lr_bias_mf]
                          + [{lam: tsr_bias_mf[lam][tsr_est] for lam in lams}
                             for tsr_est in tsr_est_types],
                          index=['cr', 'lr'] + tsr_est_types)
mf_bias_df = pd.DataFrame(mf_bias_df.stack())
mf_bias_df.index.names = ['estimator_type', 'lambda']
mf_bias_df = mf_bias_df.rename(columns={0: 'bias'})
df = pd.DataFrame({
        0.15: mf_bias_df['bias'] / gtes_norm[1]
    })

customer_type_base = {'c1':.315 }
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
for lam in lams:
    tsr_param = exp_wrapper.tsr_exp_induced_parameters(listing_types, rhos_pre_treat,
                                                       a_Cs[lam], a_Ls[lam],
                                                       customer_types, customer_proportions,
                                                       alpha, vs, lam)
    global_solution = exp_wrapper.calc_gte_from_exp_type("tsr", tau, epsilon,
                                                         **tsr_param)
    global_solutions[lam] = global_solution
    gtes[lam] = global_solution['gte']
    gtes_norm[lam] = global_solution['gte'] * norm_constants[lam]

    tsr_results = exp_wrapper.run_experiment("tsr", epsilon, tau, **tsr_param,
                                             a_C=a_Cs[lam], a_L=a_Ls[lam],
                                             customer_side_weight=cr_side_weights[lam],
                                             scale=scales[lam]
                                             )

    tsr_bias_mf[lam] = {}
    for tsr_est in tsr_est_types:
        tsr_bias_mf[lam][tsr_est] = ((tsr_results['estimator'][tsr_est] - gtes[lam])
                                     * norm_constants[lam])

    lr_exp_param = exp_wrapper.listing_side_exp_induced_parameters(listing_types, rhos_pre_treat, lr_a_L,
                                                                   customer_types,
                                                                   alpha, vs, lam, customer_proportions)
    lr_sol = exp_wrapper.run_experiment("listing", epsilon, tau, **lr_exp_param, a_L=lr_a_L)
    lr_bias_mf[lam] = abs(lr_sol['estimator']['naive_est'] - gtes[lam]) * norm_constants[lam]

    cr_exp_param = exp_wrapper.customer_side_exp_induced_parameters(listing_types, rhos_pre_treat, cr_a_C,
                                                                    customer_types, alpha, vs, lam,
                                                                    customer_proportions)
    cr_sol = exp_wrapper.run_experiment("customer", epsilon, tau, **cr_exp_param, a_C=cr_a_C)
    cr_bias_mf[lam] = abs(cr_sol['estimator']['naive_est'] - gtes[lam]) * norm_constants[lam]



mf_bias_df = pd.DataFrame([cr_bias_mf, lr_bias_mf]
                          + [{lam: tsr_bias_mf[lam][tsr_est] for lam in lams}
                             for tsr_est in tsr_est_types],
                          index=['cr', 'lr'] + tsr_est_types)
mf_bias_df = pd.DataFrame(mf_bias_df.stack())
mf_bias_df.index.names = ['estimator_type', 'lambda']
mf_bias_df = mf_bias_df.rename(columns={0: 'bias'})
df[0.32] = mf_bias_df['bias'] / gtes_norm[1]

customer_type_base = {'c1':.62 }
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
for lam in lams:
    tsr_param = exp_wrapper.tsr_exp_induced_parameters(listing_types, rhos_pre_treat,
                                                       a_Cs[lam], a_Ls[lam],
                                                       customer_types, customer_proportions,
                                                       alpha, vs, lam)
    global_solution = exp_wrapper.calc_gte_from_exp_type("tsr", tau, epsilon,
                                                         **tsr_param)
    global_solutions[lam] = global_solution
    gtes[lam] = global_solution['gte']
    gtes_norm[lam] = global_solution['gte'] * norm_constants[lam]

    tsr_results = exp_wrapper.run_experiment("tsr", epsilon, tau, **tsr_param,
                                             a_C=a_Cs[lam], a_L=a_Ls[lam],
                                             customer_side_weight=cr_side_weights[lam],
                                             scale=scales[lam]
                                             )

    tsr_bias_mf[lam] = {}
    for tsr_est in tsr_est_types:
        tsr_bias_mf[lam][tsr_est] = ((tsr_results['estimator'][tsr_est] - gtes[lam])
                                     * norm_constants[lam])

    lr_exp_param = exp_wrapper.listing_side_exp_induced_parameters(listing_types, rhos_pre_treat, lr_a_L,
                                                                   customer_types,
                                                                   alpha, vs, lam, customer_proportions)
    lr_sol = exp_wrapper.run_experiment("listing", epsilon, tau, **lr_exp_param, a_L=lr_a_L)
    lr_bias_mf[lam] = abs(lr_sol['estimator']['naive_est'] - gtes[lam]) * norm_constants[lam]

    cr_exp_param = exp_wrapper.customer_side_exp_induced_parameters(listing_types, rhos_pre_treat, cr_a_C,
                                                                    customer_types, alpha, vs, lam,
                                                                    customer_proportions)
    cr_sol = exp_wrapper.run_experiment("customer", epsilon, tau, **cr_exp_param, a_C=cr_a_C)
    cr_bias_mf[lam] = abs(cr_sol['estimator']['naive_est'] - gtes[lam]) * norm_constants[lam]



mf_bias_df = pd.DataFrame([cr_bias_mf, lr_bias_mf]
                          + [{lam: tsr_bias_mf[lam][tsr_est] for lam in lams}
                             for tsr_est in tsr_est_types],
                          index=['cr', 'lr'] + tsr_est_types)
mf_bias_df = pd.DataFrame(mf_bias_df.stack())
mf_bias_df.index.names = ['estimator_type', 'lambda']
mf_bias_df = mf_bias_df.rename(columns={0: 'bias'})
df[0.62] = mf_bias_df['bias'] / gtes_norm[1]
df = df.rename({'cr': "Customer-Side", 'lr': "Listing-Side",
                'tsri_2.0': 'TSRI-2', 'tsri_1.0': 'TSRI-1',
                'tsr_est_naive': "TSR-Naive"
                })

df = df.transpose()


rgb_dict = {'cr': (0, 114, 178), 'lr': (255, 127, 14),
                'tsrn': (0, 158, 115), 'tsri1': (240, 228, 66),
                'tsri2': (204, 121, 167),
                'cluster': (86, 180, 233)}
rgb_0_1_array = np.array(list(rgb_dict.values())) / 255
ax = plt.figure().add_subplot(111)



df.plot(ax=ax, kind='bar', color=rgb_0_1_array)
plt.xlabel("Utility without intervention")
plt.legend(loc=[1.02, 0.5])
plt.title("Mean Field Bias")
plt.ylabel("Bias/GTE")
plt.tight_layout()
sns.despine()
plt.savefig('7.1.png')
plt.show()























