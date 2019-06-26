#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import warnings
import pickle

np.seterr(all = 'raise')
warnings.filterwarnings('error')

###****************************** McGhee-von Hippel-Manning model *****************************************###

#global Na, Nb, lb, reff, alpha, vol_init, vol_injection, N_avogadro, std_volume
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


def make_np_array(nxb, nk, truncate_starting_injections):
    
    arraytype = np.dtype([ ('error'   , np.float64),\
                           ('ka0'     , np.float64),\
                           ('kb0'     , np.float64),\
                           ('theta_a' , np.float64, (nxb-truncate_starting_injections,  ) ),\
                           ('theta_b' , np.float64, (nxb-truncate_starting_injections,  ) ),\
                           ('q_per_mg', np.float64, (nxb-truncate_starting_injections-1,) ) ])
    
    return np.zeros(nk, dtype = arraytype)


def total_heat(dha, dhb, theta_a_array, theta_b_array):
    return 2*dha*np.diff(theta_a_array) + dhb*np.diff(theta_b_array)


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
                
                q_model = Na *(1 - running_avg(theta_tot)) *\
                               np.diff(theta_tot) *\
                               running_avg(np.array(chi_array)) /\
                               running_avg(xb_array[truncate_starting_injections:])

                error = relative_error(q_itc, q_model)
                                           
                if error < rel_error_max:

                    rel_error_max = error
                    
                    theta_vs_xb_mvh['error']    = 100*error 
                    theta_vs_xb_mvh['ka0']      = np.exp(ln_ka0)
                    theta_vs_xb_mvh['kb0']      = np.exp(ln_kb0)
                    theta_vs_xb_mvh['theta_a']  = theta_result[:,0]
                    theta_vs_xb_mvh['theta_b']  = theta_result[:,1] 
                    theta_vs_xb_mvh['q_per_mg'] = q_model 

    return theta_vs_xb_mvh


def cg_vs_theory(ln_Ka_array, ln_Kb_array, cg_data, reff, i = 1, error_buffer = 0.6):

    cb0, ca0, theta_mg, theta_na = cg_data.iloc[i]['conc']*std_volume/1000, 150*std_volume/1000,\
                                   cg_data.iloc[i]['mg_bound']/Nb,          cg_data.iloc[i]['na_bound']/Na

    ka_kb_combination = [(ln_ka0, ln_kb0) for ln_kb0 in ln_Kb_array for ln_ka0 in ln_Ka_array if ln_kb0 > ln_ka0]

    for ln_ka0, ln_kb0 in ka_kb_combination:

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
                continue

        except FloatingPointError:
            continue

        except Warning:
            continue

    return theta_return, np.array([theta_na, theta_mg]), lnka0_return, lnkb0_return, error_buffer

######################################################################################################################

###*************************************** ITC data retrieval *****************************************************###

def get_exp_data(exp_file, mg_conc):

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
        out_dict['conc']     = tuple( np.array([05.00, 19.10, 9.10, 0.0195]) * std_volume/1000 ) # mg_injection_conc, total_ionic_strength, nacl_ionic_strength, dpgs_conc mM

    elif mg_conc == 'mg_1_7mM':
        out_dict['xb_array'] = xb_1_7mM
        out_dict['q_array']  = q_1_7mM_per_mg
        out_dict['conc']     = tuple( np.array([09.90, 16.40, 6.40, 0.0389]) * std_volume/1000 ) # mM

    elif mg_conc == 'mg_2_5mM':
        out_dict['xb_array'] = xb_2_5mM
        out_dict['q_array']  = q_2_5mM_per_mg
        out_dict['conc']     = tuple( np.array([15.19, 14.00, 4.00, 0.0638]) * std_volume/1000 ) # mM
    
    return out_dict
###******************************************************************************************************************###

###**************************************** Model Evaluation Functions **********************************************###

def relative_error(x, y):
    return np.linalg.norm(x-y) / np.linalg.norm(x)


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

###***************************************************************************************************************************###

################################################### Execution ###################################################################

cg_data = pd.read_csv("/home/khy/google-drive/jupyter-notebooks/cg-g2-bound-counterions.csv",\
                      sep = ',', index_col = None,  engine = 'python')
ln_Ka_array = np.linspace(-10, 10, 4000)
ln_Kb_array = np.linspace(-10, 10, 4000)
reff_array  = np.linspace(1.0, 1.9, 10)

for reff in reff_array:

    var = cg_vs_theory(ln_Ka_array, ln_Kb_array, cg_data, reff,  i = 5, error_buffer = 0.7)

    file_prefix = str(cg_data.iloc[i]) + '_ka_'   + str(np.min(ln_Ka_array).astype(int)) + '_' + str(np.max(ln_Ka_array).astype(int)) +\
                             '_kb_'   + str(np.min(ln_Kb_array).astype(int)) + '_' + str(np.max(ln_Kb_array).astype(int)) +\
                             '_reff_' + str(reff) + '_error_'+ str(rel_error_max) + '.p'

    pickle_and_dump(four_param_var, "four_param_" + file_prefix)

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
                   ####get_q_vs_xb(ln_Ka_array, ln_Kb_array, exp_data['xb_array'], reff, *exp_data['conc'], truncate_starting_injections = 0)

    model_output_var[np.round(reff, 3)] = q_vs_xb_mvh

file_prefix = str(mg_conc) + '_ka_'   + str(np.min(ln_Ka_array).astype(int)) + '_' + str(np.max(ln_Ka_array).astype(int)) +\
                             '_kb_'   + str(np.min(ln_Kb_array).astype(int)) + '_' + str(np.max(ln_Kb_array).astype(int)) +\
                             '_reff_' + str(np.round(np.min(reff_array), 3)) + '_' + str(np.round(np.max(reff_array), 3)) + ".p" #\
                                            #'_inj_start_' + str(inj) + ".p", "wb")

pickle_and_dump(model_output_var, "solution_with_error_" + file_prefix)
'''


'''    

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
