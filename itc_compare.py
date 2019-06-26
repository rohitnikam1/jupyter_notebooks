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

Na            = 34
Nb            = 0.5*Na
lb            = 0.7                 # Bjerrum length (nm) for SPC/E water at 300K is 0.78
alpha         = 1.36                # eps / eps* = (T* / T)^alpha
vol_init      = 1.43                # mililitres initial titration volume
vol_injection = 8.00e-3             # mililitres injection volume
gas_constant  = 8.3144598           # J/K/mol
Temperature   = 300                 # K
std_volume    = 0.6022140857        # Standard volume L/mol
cg_box_length = 30                  # nm
cg_dpgs_conc  = 1/cg_box_length**3

cal_per_mol_to_J_per_mol  = 4.184
J_per_mol_to_kT           = gas_constant*Temperature
cal_per_mol_to_kT         = cal_per_mol_to_J_per_mol / J_per_mol_to_kT

###################################### Model Functions ##########################################################

def running_avg(x, neighbours = 2):
    return np.convolve(x, np.ones((neighbours,))/neighbours, mode='valid')


def get_kappa_from_simulation(*conc):
    mg_conc, na_conc, cl_conc = conc
    I = 0.5 * (  4*mg_conc + na_conc + cl_conc ) # num/nm3 
    k = (8 * np.pi * lb * I )**0.5
    return k


def kappa(I_num_per_nm3):
    #I_num_per_nm3 = I_mmolar * std_volume / 1000
    return (8 * np.pi * lb * I_num_per_nm3 )**0.5


def zeta(I_num_per_nm3, reff):
    return  Na * lb / reff / (1 + reff*kappa(I_num_per_nm3))


def chi(I_num_per_nm3, reff):
    return 0.5 * zeta(I_num_per_nm3, reff) * (1 + 1/(1 + reff*kappa(I_num_per_nm3))) * (alpha - 1)


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

    ln_Ka0, ln_Kb0, xb, I, ca0, cd, reff = data  # I_nacl can be given as ca0

    Ka0, Kb0, fb, fa = np.exp(ln_Ka0), np.exp(ln_Kb0), xb/Nb, ca0/Na/cd

    D  = 1 - theta[0] - theta[1]

    if mixing_entropy == True:

        yb = 4*cd*Kb0*(fb - theta[1])*D**2    -   theta[1]**2  * np.exp(-2*zeta(I, reff)*D + fb/(fb-theta[1])) * (2-theta[1])
        ya =   cd*Ka0*(fa - theta[0])*D       -   theta[0]**2  * np.exp(-  zeta(I, reff)*D + fa/(fa-theta[0]))

    else:

        yb =   cd*Kb0*(fb - theta[1])         -   theta[1]     * np.exp(-2*zeta(I, reff)*D + fb/(fb-theta[1]))
        ya =   cd*Ka0*(fa - theta[0])         -   theta[0]     * np.exp(-  zeta(I, reff)*D + fa/(fa-theta[0]))

    return ya, yb


def make_zero_cg(theta, *data, mixing_entropy = True):

    #theta[0] = theta_a
    #theta[1] = theta_b

    ln_Ka0, ln_Kb0, cb0, ca0, dpgs_conc, ionic_strength, reff = data  # I_nacl can be given as ca0

    Ka0, Kb0, fb, fa = np.exp(ln_Ka0), np.exp(ln_Kb0), cb0/Nb, ca0/Na

    D  = 1 - theta[0] - theta[1]

    if mixing_entropy == True:

        yb = 4*Kb0*(fb - dpgs_conc*theta[1])*D**2    -   theta[1]**2  * np.exp(-2*zeta(ionic_strength, reff)*D + fb/(fb-dpgs_conc*theta[1])) * (2-theta[1])
        ya =   Ka0*(fa - dpgs_conc*theta[0])*D       -   theta[0]**2  * np.exp(-  zeta(ionic_strength, reff)*D + fa/(fa-dpgs_conc*theta[0]))

    else:

        yb =   Kb0*(fb - dpgs_conc*theta[1])         -   theta[1]     * np.exp(-2*zeta(ionic_strength, reff)*D + fb/(fb-dpgs_conc*theta[1]))
        ya =   Ka0*(fa - dpgs_conc*theta[0])         -   theta[0]     * np.exp(-  zeta(ionic_strength, reff)*D + fa/(fa-dpgs_conc*theta[0]))

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



def make_np_array_trial(nxb, nk, truncate_starting_injections=0):

    arraytype = np.dtype([ ('dha'     , np.float64),\
                           ('dhb'     , np.float64),\
                           ('q_per_mg', np.float64, (nxb-truncate_starting_injections-1,) ) ])

    return np.zeros(nk, dtype = arraytype)


def make_dtype():

    arraytype = np.dtype([ ('reff'    , np.float64),\
                           ('ka0'     , np.float64),\
                           ('kb0'     , np.float64),\
                           ('dha'     , np.float64),\
                           ('dhb'     , np.float64),\
                           ('theta_a' , np.float64, (nxb-truncate_starting_injections,  ) ),\
                           ('theta_b' , np.float64, (nxb-truncate_starting_injections,  ) ),\
                           ('q_per_mg', np.float64, (nxb-truncate_starting_injections-1,) ) ])

    return arraytype



def get_q_vs_xb(ln_Ka_array, ln_Kb_array, xb_array, reff, *experimental_concentrations, truncate_starting_injections = 0):

    conc_mg_injectant, I_buffer, I_nacl, cd = np.array(experimental_concentrations) * std_volume / 1000

    nxb, nk = len(xb_array), len(ln_Ka_array)*len(ln_Kb_array)

    theta_vs_xb_mvh = make_np_array(nxb, nk, truncate_starting_injections)

    count = 0

    for ln_ka0 in ln_Ka_array:

        for ln_kb0 in ln_Kb_array:

            theta_result = np.array([0.0, 0.0])
            theta_guess  = np.array([0.0, 0.0])
            chi_array    = []

            injection_number = 0

            for injection in range(truncate_starting_injections, nxb):

                injection_number = injection + 2
                '''range starts with truncate_.. to truncate initial problematic entries for calculation.
                   1 is added because python index begins with 0. another 1 is added because first entry is omitted in experiments. So second injection corresponds to first experimental entry.'''

                I_total    = ionic_strength_itc(injection_number, I_buffer, conc_mg_injectant)
                chi_array.append(chi(I_total, reff))

                input_data = (ln_ka0, ln_kb0, xb_array[injection], I_total, I_nacl, cd, reff)  # I_nacl = ca0

                try:
                    theta_solution = fsolve(make_zero, theta_guess, args = input_data)

                    if np.all(np.logical_and(theta_solution>=0, theta_solution<=1)):
                        theta_result = np.vstack((theta_result, theta_solution))
                        theta_guess  = theta_solution
                    else:
                        break

                except FloatingPointError:
                    break

                except Warning:
                    break

            else:
                '''to be executed only when i loop terminates by exhaustion of xb_data'''
                theta_result = np.delete(theta_result, 0, axis = 0)
                theta_tot    = np.sum(theta_result, axis = 1)

                theta_vs_xb_mvh['ka0'][count]     = np.exp(ln_ka0)
                theta_vs_xb_mvh['kb0'][count]     = np.exp(ln_kb0)
                theta_vs_xb_mvh['theta_a'][count] = theta_result[:,0]
                theta_vs_xb_mvh['theta_b'][count] = theta_result[:,1]
                theta_vs_xb_mvh['q_per_mg'][count] = Na *(1 - running_avg(theta_tot)) *\
                                                    theta_vs_xb_mvh['dtheta'][count] *\
                                                    running_avg(np.array(chi_array)) / running_avg(xb_array[truncate_starting_injections:])

                count += 1

    mask            = np.array(theta_vs_xb_mvh['ka0']!= 0.0, dtype = bool)
    theta_vs_xb_mvh = theta_vs_xb_mvh[mask]
    return theta_vs_xb_mvh


def get_errorq_vs_xb(ln_Ka_array, ln_Kb_array, reff, itc_data, truncate_starting_injections = 0, rel_error_max = 0.5):

    conc_mg_injectant, I_buffer, I_nacl, cd = itc_data['conc']

    q_itc    = itc_data['q_array'][truncate_starting_injections + 1:]
    xb_array = itc_data['xb_array']

    nxb = len(xb_array)

    theta_vs_xb_mvh = make_np_array(nxb, 1, truncate_starting_injections)

    for ln_ka0 in ln_Ka_array:

        for ln_kb0 in ln_Kb_array:

            theta_result = np.array([0.0, 0.0])
            theta_guess  = np.array([0.0, 0.0])
            chi_array    = []

            injection_number = 0

            for injection in range(truncate_starting_injections, nxb):

                injection_number = injection + 2
                '''range starts with truncate_.. to truncate initial problematic entries for calculation.
                   1 is added because python index begins with 0. another 1 is added because first entry is omitted in experiments. So second injection corresponds to first experimental entry.'''

                I_total    = ionic_strength(injection_number, I_buffer, conc_mg_injectant)
                chi_array.append(chi(I_total, reff))

                input_data = (ln_ka0, ln_kb0, xb_array[injection], I_total, I_nacl, cd, reff)  # I_nacl = ca0

                try:
                    theta_solution = fsolve(make_zero, theta_guess, args = input_data)

                    if np.all(np.logical_and(theta_solution>=0, theta_solution<=1)):
                        theta_result = np.vstack((theta_result, theta_solution))
                        theta_guess  = theta_solution
                    else:
                        break

                except FloatingPointError:
                    break

                except Warning:
                    break

            else:
                '''to be executed only when i loop terminates by exhaustion of xb_data'''

                theta_result = np.delete(theta_result, 0, axis = 0)
                theta_tot    = np.sum(theta_result, axis = 1)
                dha          = running_avg(np.array(chi_array)) * (1 - running_avg(theta_tot))
                dhb          = 2 * dha
                theta_a_array      = theta_result[:,0]
                theta_b_array      = theta_result[:,1]
                q_model = total_heat_per_mgsite(dha, dhb, theta_a_array, theta_b_array, xb_array)
                '''
                q_model = Na *(1 - running_avg(theta_tot)) *\
                               np.diff(theta_tot) *\
                               running_avg(np.array(chi_array)) /\
                               running_avg(xb_array[truncate_starting_injections:])
                '''
                error = relative_error(q_itc, q_model)

                if error < rel_error_max:

                    rel_error_max = error

                    theta_vs_xb_mvh['error']    = 100*error
                    theta_vs_xb_mvh['ka0']      = np.exp(ln_ka0)
                    theta_vs_xb_mvh['kb0']      = np.exp(ln_kb0)
                    theta_vs_xb_mvh['theta_a']  = theta_a_array
                    theta_vs_xb_mvh['theta_b']  = theta_b_array
                    theta_vs_xb_mvh['q_per_mg'] = q_model

    return theta_vs_xb_mvh


def get_theta_from_K(ln_ka0, ln_kb0, *conc_data, xb_array, reff,
                     truncate_starting_injections):

    conc_mg_injectant, I_buffer, I_nacl, cd = conc_data

    for ln_ka0, ln_kb0 in ka_kb_combination:

        injection_number, theta_result, theta_guess = 0, np.array([0.0,0.0]), np.array([0.0,0.0])

        for injection in range(truncate_starting_injections, len(xb_array)):

            injection_number = injection + 2

            I_total    = ionic_strength(injection_number, I_buffer, conc_mg_injectant)

            input_data = (ln_ka0, ln_kb0, xb_array[injection], I_total, I_nacl, cd, reff)  # I_nacl = ca0

            try:
                theta_solution = fsolve(make_zero, theta_guess, args = input_data)

                if np.all(np.logical_and(theta_solution>=0, theta_solution<=1)):
                    theta_result = np.vstack((theta_result, theta_solution))
                    theta_guess  = theta_solution
                else:
                    return None

            except FloatingPointError:
                return None

            except Warning:
                return None

        else:
            '''to be executed only when i loop terminates by exhaustion of xb_data'''

            theta_result  = np.delete(theta_result, 0, axis = 0)
            yield theta_result


def get_heat_from_theta(q_itc_per_mg, ln_ka0, ln_kb0, ha_hb_combination,
                        theta_a_array, theta_b_array, count, xb_array_avg,
                        truncate_starting_injections, nk = 10000,
                        rel_error_max=0.2):

    results_mvh = make_np_array(len(xb_array_avg)+1, nk, truncate_starting_injections)

    Fa = Na*np.diff(theta_a_array) / xb_array_avg
    Fb = Nb*np.diff(theta_b_array) / xb_array_avg


    for dha, dhb in ha_hb_combination:

            q_model = dha*Fa + dhb*Fb
            error   = relative_error(q_itc_per_mg, q_model)

            if error < rel_error_max:

                results_mvh['error'][count]    = 100*error
                results_mvh['ka0'][count]      = np.exp(ln_ka0)
                results_mvh['kb0'][count]      = np.exp(ln_kb0)
                results_mvh['dha'][count]      = dha
                results_mvh['dhb'][count]      = dhb
                results_mvh['theta_a'][count]  = theta_a_array
                results_mvh['theta_b'][count]  = theta_b_array
                results_mvh['q_per_mg_mgsite'][count] = q_model

                count += 1

            else:
                continue
    else:
        mask        = np.array(results_mvh['ka0']!= 0.0, dtype = bool)
        return results_mvh[mask]


def comb_ka_kb(x_array, y_array):
    return [(x,y) for x in x_array for y in y_array if y>x]


def four_param_trial(ln_Ka_array, ln_Kb_array, dha_array, dhb_array, itc_data,
                     reff, truncate_starting_injections = 0, rel_error_max=0.2):

    q_itc_per_mg      = itc_data['q_array'][truncate_starting_injections + 1:]
    xb_array_raw_data          = itc_data['xb_array']
    xb_array          = xb_array_raw_data #np.linspace(xb_array_raw_data[0], xb_array_raw_data[-1], 70)
    xb_array_avg      = running_avg(xb_array[truncate_starting_injections:])
    ka_kb_combination = comb_ka_kb(ln_Ka_array, ln_Kb_array)
    ha_hb_combination = [(dha, dhb) for dha in dha_array for dhb in dhb_array if dhb != dha]
    count = 0

    while i < len(ka_kb_combination):

            result_theta = get_theta_from_K(ln_ka0, ln_kb0, *itc_data['conc'],
                                            xb_array=xb_array, reff=reff,
                                            truncate_starting_injections=truncate_starting_injections)

            if result_theta is not None:

                result_heat = get_heat_from_theta(q_itc_per_mg, ln_ka0, ln_kb0,
                                                  ha_hb_combination,
                                                  result_theta[:,0],
                                                  result_theta[:,1],
                                                  count,
                                                  xb_array_avg = xb_array_avg,
                                                  truncate_starting_injections=truncate_starting_injections,
                                                  nk = 10000,
                                                  rel_error_max=rel_error_max)

    return result_heat


def make_dha_dhb_comb(dha_array, dhb_array, theta_a_array, theta_b_array):

    factor = 2*np.max(np.abs(np.diff(theta_a_array)/np.diff(theta_b_array)))

    return [(dha, dhb) for dhb in dhb_array for dha in dha_array if dhb > dha*factor]


def total_heat_per_mg(dh, theta_array, xb_array, truncate_starting_injections = 0):
    return (Na*dh[0]*np.diff(theta_array[:,0]) + Nb*dh[1]*np.diff(theta_array[:,1])) / running_avg(xb_array[truncate_starting_injections:])


def total_heat_per_mg_trial(dh, xb_array, truncate_starting_injections = 0):
    return (Na*dh[0] + Nb*dh[1]) / running_avg(xb_array[truncate_starting_injections:])


def relative_error(x, y):
    return np.linalg.norm(x-y) / np.linalg.norm(x)


def heat_error(dh, *data):

    #dha = dh[0]
    #dhb = dh[1]

    theta_result, xb_array, q_itc_per_mg = data

    q_model = total_heat_per_mg(dh, theta_result, xb_array)
    error   = relative_error(q_itc_per_mg, q_model)

    return error


def heat_error_trial(dh, *data):

    #dha = dh[0]
    #dhb = dh[1]

    xb_array, q_itc_per_mg = data

    q_model = total_heat_per_mg_trial(dh, xb_array)
    error   = relative_error(q_itc_per_mg, q_model)

    return error


def get_dh_trial(q_itc_per_mg, xb_array, tolerance = 0.0001):
    
    data, dh_guess = (xb_array, q_itc_per_mg), np.array([0.0, 0.0])
    sol            = minimize(heat_error_trial, dh_guess, args = data, tol = tolerance)
    dh             = sol.x

    return dh, total_heat_per_mg_trial(dh, xb_array)
    

def get_dh_four_param_fit(ln_Ka_array, ln_Kb_array, itc_data, reff,
                          truncate_starting_injections = 0, tolerance = 0.001):

    #conc_mg_injectant, I_buffer, I_nacl, cd = itc_data['conc']

    q_itc_per_mg, xb_array, conc_mg_injectant, I_buffer, I_nacl, cd = itc_data

    ka_kb_combination = [(ln_ka0, ln_kb0) for ln_kb0 in ln_Kb_array for ln_ka0 in ln_Ka_array]

    results_mvh = make_np_array(len(xb_array), 1, truncate_starting_injections)

    for ln_ka0, ln_kb0 in ka_kb_combination:

            theta_result = np.array([0.0, 0.0])
            theta_guess  = np.array([0.0, 0.0])

            injection_number = 0

            for injection in range(truncate_starting_injections, len(xb_array)):

                injection_number = injection + 2

                I_total    = ionic_strength(injection_number, I_buffer, conc_mg_injectant)

                input_data = (ln_ka0, ln_kb0, xb_array[injection], I_total, I_nacl, cd, reff)  # I_nacl = ca0

                try:
                    theta_solution = fsolve(make_zero, theta_guess, args = input_data)

                    if np.all(np.logical_and(theta_solution>=0, theta_solution<=1)):
                        theta_result = np.vstack((theta_result, theta_solution))
                        theta_guess  = theta_solution
                    else:
                        break

                except FloatingPointError:
                    break

                except Warning:
                    break

            else:
                theta_result   = np.delete(theta_result, 0, axis = 0)
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


def get_errorq_four_param_fit(ln_Ka_array, ln_Kb_array, dha_array, dhb_array,
                              itc_data, reff, truncate_starting_injections = 0,
                              rel_error_max = 0.2):

    conc_mg_injectant, I_buffer, I_nacl, cd = itc_data['conc']

    q_itc_per_mg      = itc_data['q_array'][truncate_starting_injections + 1:]
    xb_array          = itc_data['xb_array']
    ka_kb_combination = [(ln_ka0, ln_kb0) for ln_kb0 in ln_Kb_array for ln_ka0 in ln_Ka_array]

    results_mvh = make_np_array(len(xb_array), 1, truncate_starting_injections)

    for ln_ka0, ln_kb0 in ka_kb_combination:

            theta_result = np.array([0.0, 0.0])
            theta_guess  = np.array([0.0, 0.0])

            injection_number = 0

            for injection in range(truncate_starting_injections, len(xb_array)):

                injection_number = injection + 2

                I_total    = ionic_strength(injection_number, I_buffer, conc_mg_injectant)

                input_data = (ln_ka0, ln_kb0, xb_array[injection], I_total, I_nacl, cd, reff)  # I_nacl = ca0

                try:
                    theta_solution = fsolve(make_zero, theta_guess, args = input_data)

                    if np.all(np.logical_and(theta_solution>=0, theta_solution<=1)):
                        theta_result = np.vstack((theta_result, theta_solution))
                        theta_guess  = theta_solution
                    else:
                        break

                except FloatingPointError:
                    break

                except Warning:
                    break

            else:
                '''to be executed only when i loop terminates by exhaustion of xb_data'''

                theta_result  = np.delete(theta_result, 0, axis = 0)

                ha_hb_combination = make_dha_dhb_comb(dha_array, dhb_array, theta_result[:,0], theta_result[:,1])

                for dha, dhb in ha_hb_combination:

                    q_model = total_heat_per_mgsite(dha, dhb, theta_result, xb_array)
                    error   = relative_error(q_itc_per_mg, q_model)

                    if error < rel_error_max:

                        new_result = np.array([(100*error, np.exp(ln_ka0),
                                                np.exp(ln_kb0), dha, dhb,
                                                theta_result[:,0],
                                                theta_result[:,1], q_model)],
                                              dtype = results_mvh.dtype)

                        results_mvh = np.append(results_mvh, new_result)

                        '''

                        results_mvh['error'][count]    = 100*error
                        results_mvh['ka0'][count]      = np.exp(ln_ka0)
                        results_mvh['kb0'][count]      = np.exp(ln_kb0)
                        results_mvh['dha'][count]      = dha
                        results_mvh['dhb'][count]      = dhb
                        results_mvh['theta_a'][count]  = theta_result[:,0]
                        results_mvh['theta_b'][count]  = theta_result[:,1]
                        results_mvh['q_per_mg_mgsite'][count] = q_model

                        count += 1
                        '''

    mask            = np.array(results_mvh['ka0']!= 0.0, dtype = bool)
    results_mvh = results_mvh[mask]
    return results_mvh


def cg_vs_theory(ln_Ka_array, ln_Kb_array, cg_data, reff, i = 1, error_buffer = 0.6):

    cb0, ca0, theta_mg, theta_na = cg_data.iloc[i]['conc']*std_volume/1000, 150*std_volume/1000,\
                                   cg_data.iloc[i]['mg_bound']/Nb,          cg_data.iloc[i]['na_bound']/Na

    for ln_ka0 in ln_Ka_array:

        for ln_kb0 in ln_Kb_array:

            theta_guess  = np.array([0.0, 0.0])

            ionic_strength = 2*cb0 + ca0

            input_data = (ln_ka0, ln_kb0, cb0, ca0, cg_dpgs_conc, ionic_strength, reff)  # I_nacl = ca0

            try:
                theta_solution = fsolve(make_zero_cg, theta_guess, args = input_data)

                if np.all(np.logical_and(theta_solution>=0, theta_solution<=1)):

                    error = relative_error(theta_solution, np.array([theta_na, theta_mg]))

                    if error < error_buffer:
                        error_buffer = error
                        theta_return, lnka0_return, lnkb0_return = theta_solution, ln_ka0, ln_kb0

                else:
                    break

            except FloatingPointError:
                break

            except Warning:
                break

    return reff, theta_return, np.array([theta_na, theta_mg]), lnka0_return, lnkb0_return, error_buffer

######################################################################################################################

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
        out_dict['conc']     = tuple(np.array([05.00, 19.10, 9.10, 0.0195]) * std_volume/1000 ) # mg_injection_conc, total_ionic_strength, nacl_ionic_strength, dpgs_conc mM

    elif mg_conc == 'mg_1_7mM':
        out_dict['xb_array'] = xb_1_7mM
        out_dict['q_array']  = q_1_7mM_per_mg
        out_dict['conc']     = tuple(np.array([09.90, 16.40, 6.40, 0.0389]) * std_volume/1000 ) # mM

    elif mg_conc == 'mg_2_5mM':
        out_dict['xb_array'] = xb_2_5mM
        out_dict['q_array']  = q_2_5mM_per_mg
        out_dict['conc']     = tuple(np.array([15.19, 14.00, 4.00, 0.0638]) * std_volume/1000 ) # mM

    return out_dict['q_array'][truncate_starting_injections + 1:], out_dict['xb_array'], out_dict['conc'][0], out_dict['conc'][1], out_dict['conc'][2], out_dict['conc'][3]
###******************************************************************************************************************###

###**************************************** Model Evaluation Functions **********************************************###


def relative_error_scalar(x, y):
    return np.abs(x-y) / np.abs(x)

def get_index(np_array, element):
    return np_array.tolist().index(element)


def open_and_assign(file_name):
    t    = open(file_name, "rb")
    var = pickle.load(t)
    t.close()
    return var

def pickle_and_dump(var, file_name):
    pickle_out = open(file_name, "wb")
    pickle.dump(var, pickle_out)
    pickle_out.close()

'''
def evaluate_error(model_output_var, exp_data, xb_start = 0):

    xb_array_full = exp_data['xb_array']
    q_exp_full    = exp_data['q_array']

    nxb = len(xb_array_full)

    min_sum_sqr_error_dict = {}
    ka0_dict = {}
    kb0_dict = {}
    k_comb_index_dict = {}

    xb_input      = xb_array_full[xb_start:]
    q_exp_compare = q_exp_full[xb_start + 1:]#running_avg(q_exp_full)

   for reff in model_output_var.keys():

        error_array = np.zeros(len(model_output_var[reff]))

        for entry in range(len(error_array)):

            error              = relative_error(q_exp_compare, model_output_var[reff]['q_per_mg'][entry])
            error_array[entry] = error*100

        k_combination_index = get_index(error_array, np.min(error_array))
        ka0                 = model_output_var[reff]['ka0'][k_combination_index]
        kb0                 = model_output_var[reff]['kb0'][k_combination_index]

        min_sum_sqr_error_dict[reff] = np.round(np.min(error_array), 5)
        ka0_dict[reff]               = np.log(ka0)
        kb0_dict[reff]               = np.log(kb0)
        k_comb_index_dict[reff]      = k_combination_index

    min_error              = min(min_sum_sqr_error_dict.values())
    min_error_reff         = min(min_sum_sqr_error_dict, key = min_sum_sqr_error_dict.get)
    min_error_k_comb_index = k_comb_index_dict[min_error_reff]

    arraytype = np.dtype([ ('rel_error', np.float64),\
                           ('ka0'      , np.float64),\
                           ('kb0'      , np.float64),\
                           ('theta_a'  , np.float64, (nxb,  ) ),\
                           ('theta_b'  , np.float64, (nxb,  ) ),\
                           ('q_per_mg' , np.float64, (nxb-1,) ) ])

    solution = np.zeros(1, dtype = arraytype)

    solution['rel_error'] = min_error
    solution['ka0']       = ka0_dict[min_error_reff]
    solution['kb0']       = kb0_dict[min_error_reff]
    solution['theta_a']   = model_output_var[min_error_reff]['theta_a'][min_error_k_comb_index]
    solution['theta_b']   = model_output_var[min_error_reff]['theta_b'][min_error_k_comb_index]
    solution['q_per_mg']  = model_output_var[min_error_reff]['q_per_mg'][min_error_k_comb_index]

    error_series = pd.Series(min_sum_sqr_error_dict)
    ka0_series   = pd.Series(ka0_dict)
    kb0_series   = pd.Series(kb0_dict)

    dataframe = pd.DataFrame( {'error': error_series, 'ka0': ka0_series, 'kb0': kb0_series}, columns = ['error', 'ka0', 'kb0'] )

    return solution, dataframe
'''
###***************************************************************************************************************************###

################################################### Execution ###################################################################

'''
################################################## CG ##################################################
cg_data = pd.read_csv("/home/khy/google-drive/jupyter-notebooks/cg-g2-bound-counterions.csv",\
                      sep = ',', index_col = None,  engine = 'python')

ln_Ka_array = np.linspace(-10, 10, 4000)
ln_Kb_array = np.linspace(-10, 10, 4000)
reff_array  = np.linspace(1.0, 1.9, 10)

#for reff in reff_array:

print(cg_vs_theory(ln_Ka_array, ln_Kb_array, cg_data, 1.6,  i = 4, error_buffer = 0.7))
'''

#################################################### Four Parameters ####################################################

mg_conc      = 'mg_1_7mM'
#exp_data     = get_exp_data('/home/rohit/google-drive/magnesium/itc/data.csv', mg_conc)
exp_data    = get_exp_data('/home/khy/google-drive/magnesium/itc/data.csv', mg_conc)
ln_Ka_array  = np.linspace(-5, 10, 500)
ln_Kb_array  = np.linspace(-5, 10, 500)
reff_array   = np.linspace(1.3, 1.9, 7)
#dha_array   = np.linspace(-10, 10, 200) # chi(14*std_volume/1000, reff)*
#dhb_array   = np.linspace(-10, 10, 200)

'''
q_itc_per_mg, xb_array = exp_data[0], exp_data[1]
dh, q = get_dh_trial(q_itc_per_mg, xb_array)
print(dh)
plt.plot(running_avg(xb_array), q, running_avg(xb_array), q_itc_per_mg)
plt.show()
'''

results_mvh = make_np_array(len(exp_data[1]), 1)

results_mvh = np.append(results_mvh, get_dh_four_param_fit(ln_Ka_array,
                                                           ln_Kb_array,
                                                           exp_data, reff=2.1))

'''
for reff in reff_array:

   results_mvh = np.append(results_mvh, get_dh_four_param_fit(ln_Ka_array,
                                                           ln_Kb_array,
                                                           exp_data, reff=2.1))
'''
results_mvh = np.delete(results_mvh, 0)

file_prefix = 'itc_'+ str(mg_conc) + '_ka_'   + str(np.min(ln_Ka_array).astype(int)) + '_' + str(np.max(ln_Ka_array).astype(int)) +\
                             '_kb_' + str(np.min(ln_Kb_array).astype(int)) + '_' + str(np.max(ln_Kb_array).astype(int)) + '.p'

pickle_and_dump(results_mvh, file_prefix + '_tol_0.001')

'''
four_param_var = get_errorq_four_param_fit(ln_Ka_array, ln_Kb_array, dha_array,
                                           dhb_array, exp_data, reff,
                                           rel_error_max = 0.2)

file_prefix = 'itc_'+ str(mg_conc) + '_ka_'   + str(np.min(ln_Ka_array).astype(int)) + '_' + str(np.max(ln_Ka_array).astype(int)) +\
                             '_kb_'   + str(np.min(ln_Kb_array).astype(int)) + '_' + str(np.max(ln_Kb_array).astype(int)) +\
                             '_dha_'  + str(np.min(dha_array).astype(int))   + '_' + str(np.max(dha_array).astype(int))   +\
                             '_dhb_'  + str(np.min(dhb_array).astype(int))   + '_' + str(np.max(dhb_array).astype(int))   +\
                             '_reff_' + str(reff) + '_error_'+ str(rel_error_max) + '.p'

'''
##################################################### Two parameters ##################################################
'''
mg_conc             = 'mg_2_5mM'
exp_data            = get_exp_data('/home/khy/google-drive/magnesium/itc/data.csv', mg_conc) # /home/rohit/google-drive/magnesium/itc/data.csv
ln_Kb_array         = np.linspace(-5, 25, 100)
ln_Ka_array         = np.linspace(-9, 0, 100)
reff_array          = np.linspace(0.7, 1.8, 30)
#starting_inj_array  = range(1)

model_output_var = {}

for reff in reff_array:

    q_vs_xb_mvh  = get_errorq_vs_xb(ln_Ka_array, ln_Kb_array, reff, exp_data, truncate_starting_injections = 0)

    model_output_var[np.round(reff, 3)] = q_vs_xb_mvh

file_prefix = str(mg_conc) + '_ka_'   + str(np.min(ln_Ka_array).astype(int)) + '_' + str(np.max(ln_Ka_array).astype(int)) +\
                             '_kb_'   + str(np.min(ln_Kb_array).astype(int)) + '_' + str(np.max(ln_Kb_array).astype(int)) +\
                             '_reff_' + str(np.round(np.min(reff_array), 3)) + '_' + str(np.round(np.max(reff_array), 3)) + ".p" #\
                                            #'_inj_start_' + str(inj) + ".p", "wb")

pickle_and_dump(model_output_var, "solution_with_error_" + file_prefix)





for reff in reff_array:

    q_vs_xb_mvh  = get_q_vs_xb(ln_Ka_array, ln_Kb_array, exp_data['xb_array'], reff, *exp_data['conc'], truncate_starting_injections = 0)

    model_output_var[np.round(reff, 3)] = q_vs_xb_mvh

solution = evaluate_error(model_output_var, exp_data, xb_start = 0)

file_prefix = str(mg_conc) + '_ka_'   + str(np.min(ln_Ka_array).astype(int)) + '_' + str(np.max(ln_Ka_array).astype(int)) +\
                             '_kb_'   + str(np.min(ln_Kb_array).astype(int)) + '_' + str(np.max(ln_Kb_array).astype(int)) +\
                             '_reff_' + str(np.round(np.min(reff_array), 3)) + '_' + str(np.round(np.max(reff_array), 3)) + ".p" #\
                                            #'_inj_start_' + str(inj) + ".p", "wb")

pickle_and_dump(model_output_var, "model_output_" + file_prefix)
pickle_and_dump(solution        , "solution_"     + file_prefix)
'''
'''
The tuning parameters are
---curtaining the initial entries in xb_array
---changing the ranges of K. using different ranges for KA0 and KB0
---radius of dPGS
---without mixing entropy
---first/last chi element, chi array

'''
# increasing dPGS concentration makes point of inflexion more evident
