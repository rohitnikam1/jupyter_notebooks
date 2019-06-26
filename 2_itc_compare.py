#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
import pandas as pd
import warnings
import pickle
np.seterr(all = 'raise')
warnings.filterwarnings('error')

###****************************** McGhee-von Hippel-Manning model *****************************************###

Na            = 19.3
lb            = 0.7                 # Bjerrum length (nm) for SPC/E water at 300K is 0.78
alpha         = 1.36                # eps / eps* = (T* / T)^alpha
vol_init      = 1.43                # mililitres initial titration volume
vol_injection = 8.00e-3             # mililitres injection volume
gas_constant  = 8.3144598           # J/K/mol
Temperature   = 300                 # K
std_volume    = 0.6022140857        # Standard volume L/mol
tolerance     = 0.1                 # heat tolerance
cg_box_length = 30                  # nm
cg_dpgs_conc  = 1/cg_box_length**3

cal_per_mol_to_J_per_mol  = 4.184
J_per_mol_to_kT           = gas_constant*Temperature
cal_per_mol_to_kT         = cal_per_mol_to_J_per_mol / J_per_mol_to_kT

###################################### Model Functions ##########################################################

def running_avg(x, neighbours = 2):
    return np.convolve(x, np.ones((neighbours,))/neighbours, mode='valid')

def relative_error(x, y):
    return np.linalg.norm(x-y) / np.linalg.norm(x)

def kappa(I_num_per_nm3):
    #I_num_per_nm3 = I_mmolar * std_volume / 1000
    return (8 * np.pi * lb * I_num_per_nm3 )**0.5


def zeta(I_num_per_nm3, reff, Na):
    return  Na * lb / reff / (1 + reff*kappa(I_num_per_nm3))


def chi(I_num_per_nm3, reff, Na):
    return 0.5 * zeta(I_num_per_nm3, reff, Na) * (1 + 1/(1 + reff*kappa(I_num_per_nm3))) * (alpha - 1)


def ionic_strength(injection_number, I_buffer, conc_mg_injectant):
    '''
    I_buffer = combined ionic strength of MOPS and NaCl
    I_nacl   = ionic strength of NaCl
    '''
    conc_mg_solution = conc_mg_injectant*injection_number*vol_injection / (vol_init + injection_number*vol_injection)

    I_mgcl2 = 3*conc_mg_solution

    I_total = I_buffer + I_mgcl2

    return I_total


def make_zero(theta, *data, mixing_entropy = True):

    #theta[0] = theta_a
    #theta[1] = theta_b

    ln_Ka0, ln_Kb0, xb, I, ca0, cd, reff, Na = data  # I_nacl can be given as ca0

    Ka0, Kb0, fb, fa = np.exp(ln_Ka0), np.exp(ln_Kb0), 2*xb/Na, ca0/Na/cd

    D  = 1 - theta[0] - theta[1]

    if mixing_entropy == True:

        yb = 4*cd*Kb0*(fb - theta[1])*D**2    -   theta[1]**2  * np.exp(-2*zeta(I, reff, Na)*D + fb/(fb-theta[1])) * (2-theta[1])
        ya =   cd*Ka0*(fa - theta[0])*D       -   theta[0]**2  * np.exp(-  zeta(I, reff, Na)*D + fa/(fa-theta[0]))

    else:

        yb =   cd*Kb0*(fb - theta[1])         -   theta[1]     * np.exp(-2*zeta(I, reff, Na)*D + fb/(fb-theta[1]))
        ya =   cd*Ka0*(fa - theta[0])         -   theta[0]     * np.exp(-zeta(I, reff, Na)*D + fa/(fa-theta[0]))

    return ya, yb



def make_np_array(nxb, nk, truncate_starting_injections=0):

    arraytype = np.dtype([ ('reff'    , np.float64),\
                           ('ka0'     , np.float64),\
                           ('kb0'     , np.float64),\
                           ('dha'     , np.float64),\
                           ('dhb'     , np.float64),\
                           ('theta_a' , np.float64, (nxb-truncate_starting_injections,  ) ),\
                           ('theta_b' , np.float64, (nxb-truncate_starting_injections,  ) ),\
                           ('q_per_mg', np.float64, (nxb-truncate_starting_injections-1,) ) ])

    return np.zeros(nk, dtype = arraytype)

def make_np_array_q(nxb, nk, truncate_starting_injections=0):

    arraytype = np.dtype([ ('reff'    , np.float64),\
                           ('ka0'     , np.float64),\
                           ('kb0'     , np.float64),\
                           ('theta_a' , np.float64, (nxb-truncate_starting_injections,  ) ),\
                           ('theta_b' , np.float64, (nxb-truncate_starting_injections,  ) ),\
                           ('q_per_mg', np.float64, (nxb-truncate_starting_injections-1,) ) ])

    return np.zeros(nk, dtype = arraytype)

def total_heat_per_mg(dh, theta_array, xb_array, truncate_starting_injections = 0):
    return (Na*dh[0]*np.diff(theta_array[:,0]) + Nb*dh[1]*np.diff(theta_array[:,1])) / running_avg(xb_array[truncate_starting_injections:])


def get_dq(Na, theta, theta_array, chi, xb):
    return Na*chi*(1-np.sum(theta))*(np.sum(theta) - np.sum(theta_array[-1]))/xb


def heat_error(dh, *data):

    #dha = dh[0]
    #dhb = dh[1]

    theta_result, xb_array, q_itc_per_mg = data

    q_model = total_heat_per_mg(dh, theta_result, xb_array)
    error   = relative_error(q_itc_per_mg, q_model)

    return error



def get_q(ln_Ka_array, ln_Kb_array, itc_data, reff, truncate_starting_injections = 0):

    q_itc_per_mg, xb_array, conc_mg_injectant, I_buffer, I_nacl, cd = itc_data
    print(len(q_itc_per_mg))

    nmg_inj = conc_mg_injectant * vol_injection

    n_dpgs = cd * vol_init

    ka_kb_combination = [(ln_ka0, ln_kb0) for ln_kb0 in ln_Kb_array for ln_ka0 in ln_Ka_array]

    results_mvh = make_np_array_q(len(xb_array), 1, truncate_starting_injections)

    for ln_ka0, ln_kb0 in ka_kb_combination:

            Na           = 19.3
            theta_result = np.array([[0.0, 0.0], [Na/34, 0.0]])
            theta_guess  = np.array([0.0, 0.0])
            dq           = []
            vol          = vol_init + (truncate_starting_injections) * vol_injection
            mg_conc      = 0 #nmg_inj / vol

            for injection in xb_array[truncate_starting_injections:]:

                vol     += vol_injection
                mg_conc += nmg_inj / vol
                cd       = n_dpgs / vol
                I_total  = I_buffer + 3*mg_conc

                input_data = (ln_ka0, ln_kb0, injection, I_total, I_nacl, cd, reff, Na)  # I_nacl = ca0

                try:
                    theta_solution = fsolve(make_zero, theta_guess, args = input_data)

                    if np.sum(theta_solution) < 1:

                        mg_conc        -= 0.5*Na*cd*theta_solution[1]
                        I_total         = I_buffer + 3*mg_conc
                        dq.append(get_dq(Na, theta_solution, theta_result, chi(I_total, reff, Na), injection))
                        bound_ions      = Na*theta_solution
                        theta_result    = np.vstack((theta_result, bound_ions/34))
                        Na             -= Na*np.sum(theta_solution)
                        theta_guess     = bound_ions/Na


                    elif Na > 0:

                        theta_solution /= np.sum(theta_solution)
                        mg_conc        -= 0.5*Na*cd*theta_solution[1]
                        I_total         = I_buffer + 3*mg_conc
                        dq.append(get_dq(Na, theta_solution, theta_result, chi(I_total, reff, Na), injection))
                        bound_ions      = Na*theta_solution
                        theta_result    = np.vstack((theta_result, bound_ions/34))
                        theta_guess     = np.array([0.0, 0.0])
                        Na              = 0

                    else:
                        theta_result    = np.vstack((theta_result, np.array([0.0, 0.0])))
                        theta_guess     = np.array([0.0, 0.0])
                        dq.append(0)

                except FloatingPointError:
                    break

                except Warning:
                    break

            else:
                theta_result = np.delete(theta_result, 0)
                dq = np.array(dq)
                error = relative_error(q_itc_per_mg, dq)

                if error < tolerance:

                    new_result = np.array([(reff, np.exp(ln_ka0), np.exp(ln_kb0),
                                            theta_result[:,0],
                                            theta_result[:,1], dq)],
                                            dtype = results_mvh.dtype)

                    results_mvh = np.append(results_mvh, new_result)

    results_mvh = np.delete(results_mvh, 0)
    return results_mvh


def get_dh_four_param_fit(ln_Ka_array, ln_Kb_array, itc_data, reff, truncate_starting_injections = 0):

    q_itc_per_mg, xb_array, conc_mg_injectant, I_buffer, I_nacl, cd = itc_data

    nmg_inj = conc_mg_injectant * vol_injection

    n_dpgs = cd * vol_init

    ka_kb_combination = [(ln_ka0, ln_kb0) for ln_kb0 in ln_Kb_array for ln_ka0 in ln_Ka_array]

    results_mvh = make_np_array(len(xb_array), 1, truncate_starting_injections)

    for ln_ka0, ln_kb0 in ka_kb_combination:

            theta_result = np.array([Na/34, 0.0])
            theta_guess  = np.array([0.0, 0.0])

            vol = vol_init + truncate_starting_injections * vol_injection

            for injection in xb_array[truncate_starting_injections:]:

                vol     += vol_injection
                mg_conc += nmg_inj / vol
                cd       = n_dpgs / vol
                I_total  = I_buffer + 3*mg_conc

                input_data = (ln_ka0, ln_kb0, injection, I_total, I_nacl, cd, reff, Na)  # I_nacl = ca0

                try:
                    theta_solution = fsolve(make_zero, theta_guess, args = input_data)

                    if np.all(np.logical_and(theta_solution>=0, theta_solution<=1)):

                        if np.sum(theta_solution) < 1:

                            bound_ions   = Na*theta_solution
                            theta_result = np.vstack((theta_result, bound_ions/34))
                            mg_conc     -= 0.5*Na*cd*theta_solution[1]
                            Na          -= Na*np.sum(theta_solution)
                            theta_guess  = bound_ions/Na

                        elif Na > 0:

                            theta_solution /= np.sum(theta_solution)
                            bound_ions      = Na*theta_solution
                            theta_result    = np.vstack((theta_result, bound_ions/34))
                            mg_conc        -= 0.5*Na*cd*theta_solution[1]
                            Na              = 0
                            theta_guess     = np.array([0.0, 0.0])

                        else:
                            theta_result = np.vstack((theta_result, np.array([0.0, 0.0])))
                    else:
                        break

                except FloatingPointError:
                    break

                except Warning:
                    break

            else:
                data, dh_guess = (theta_result, xb_array, q_itc_per_mg), np.array([0.0, 0.0])
                sol            = minimize(heat_error, dh_guess, args = data, tol = tolerance)
                dh             = sol.x

                if np.all(np.logical_and(np.abs(dh) < 10, dh != 0.0)):

                    new_result = np.array([(reff, np.exp(ln_ka0), np.exp(ln_kb0), dh[0], dh[1],
                                            theta_result[:,0], theta_result[:,1], total_heat_per_mg(dh, theta_result, xb_array))],
                                            dtype = results_mvh.dtype)

                    results_mvh = np.append(results_mvh, new_result)

    results_mvh = np.delete(results_mvh, 0)
    return results_mvh


###*************************************** ITC data retrieval *****************************************************###

def get_exp_data(exp_file, mg_conc, truncate_starting_injections = 0):

    itc_data  = pd.read_csv(exp_file, sep = ',', index_col = None,  engine = 'python')

    xb_0_8mM, xb_1_7mM, xb_2_5mM = np.array(itc_data["ratio_0_8mM"].dropna()),\
                                   np.array(itc_data["ratio_1_7mM"]),\
                                   np.array(itc_data["ratio_2_5mM"])

    q_0_8mM, q_1_7mM, q_2_5mM    = np.array(itc_data["q_0_8mM"].dropna()) * cal_per_mol_to_J_per_mol,\
                                   np.array(itc_data["q_1_7mM"]) * cal_per_mol_to_J_per_mol,\
                                   np.array(itc_data["q_2_5mM"]) * cal_per_mol_to_J_per_mol

    q_0_8mM_per_mg, q_1_7mM_per_mg, q_2_5mM_per_mg = q_0_8mM / (gas_constant*Temperature),\
                                                     q_1_7mM / (gas_constant*Temperature),\
                                                     q_2_5mM / (gas_constant*Temperature)

    q_0_8mM_per_d, q_1_7mM_per_d, q_2_5mM_per_d    = q_0_8mM_per_mg * xb_0_8mM ,\
                                                     q_1_7mM_per_mg * xb_1_7mM ,\
                                                     q_2_5mM_per_mg * xb_2_5mM
    out_dict = {}

    if mg_conc == 'mg_0_8mM':
        out_dict['xb_array'] = xb_0_8mM
        out_dict['q_array']  = q_0_8mM_per_mg
        out_dict['conc']     = tuple(np.array([05.00, 19.10, 9.10, 0.0195]) *
                                     std_volume/1000 )
        # mg_injection_conc, total_ionic_strength, nacl_ionic_strength, dpgs_conc (number per nm3)

    elif mg_conc == 'mg_1_7mM':
        out_dict['xb_array'] = xb_1_7mM
        out_dict['q_array']  = q_1_7mM_per_mg
        out_dict['conc']     = tuple(np.array([09.90, 16.40, 6.40, 0.0389]) *
                                     std_volume/1000 ) # number per nm3

    elif mg_conc == 'mg_2_5mM':
        out_dict['xb_array'] = xb_2_5mM
        out_dict['q_array']  = q_2_5mM_per_mg
        out_dict['conc']     = tuple(np.array([15.19, 14.00, 4.00, 0.0638]) * std_volume/1000 )

    return out_dict['q_array'][truncate_starting_injections:], out_dict['xb_array'], out_dict['conc'][0], out_dict['conc'][1], out_dict['conc'][2], out_dict['conc'][3]

###******************************************************************************************************************###

def pickle_and_dump(var, file_name):
    pickle_out = open(file_name, "wb")
    pickle.dump(var, pickle_out)
    pickle_out.close()

################################################### Execution ###################################################################
mg_conc      = 'mg_2_5mM'
exp_data     = get_exp_data('/home/khy/google-drive/magnesium/itc/data.csv', mg_conc)
print(len(exp_data[0]))
ln_Ka_array  = np.linspace(-5, 10, 500)
ln_Kb_array  = np.linspace(-5, 10, 500)
reff_array   = np.linspace(1.6, 2.1, 6)


results_mvh = make_np_array_q(len(exp_data[1]), 1)

for reff in reff_array:

    results_mvh = np.append(results_mvh, get_q(ln_Ka_array, ln_Kb_array, exp_data, reff))
    results_mvh = np.delete(results_mvh, 0)

file_prefix = 'itc_'+ str(mg_conc) + '_ka_'   + str(np.min(ln_Ka_array).astype(int)) + '_' + str(np.max(ln_Ka_array).astype(int)) +\
                             '_kb_' + str(np.min(ln_Kb_array).astype(int)) +'_' + str(np.max(ln_Kb_array).astype(int)) + '_tol_0.1_.p'

pickle_and_dump(results_mvh, file_prefix)
