#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import warnings
import pickle
import time

np.seterr(all = 'raise')
warnings.filterwarnings('error')

###****************************** McGhee-von Hippel-Manning model *****************************************###

#global Na, Nb, lb, reff, alpha, vol_init, vol_injection, N_avogadro, std_volume
Na            = 24
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

def kappa(I_num_per_nm3):
    #I_num_per_nm3 = I_mmolar * std_volume / 1000
    return (8 * np.pi * lb * I_num_per_nm3 )**0.5 


def zeta(I_num_per_nm3, reff):
    return  Na * lb / reff / (1 + reff*kappa(I_num_per_nm3)) 

def relative_error(x, y):
    return np.linalg.norm(x-y) / np.linalg.norm(x)

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



def make_cg_array():
    
    arraytype = np.dtype([ ('reff'    , np.float64),\
                           ('error'   , np.float64),\
                           ('ka'      , np.float64),\
                           ('kb'      , np.float64),\
                           ('theta_a' , np.float64),\
                           ('theta_b' , np.float64)])
    
    return np.zeros(1, dtype = arraytype)


def cg_vs_theory(ln_Ka_array, ln_Kb_array, cg_data, reff, i, rel_error_max):
    
    cb0, ca0, theta_mg, theta_na = cg_data.iloc[i]['conc']*std_volume/1000, 150*std_volume/1000,\
                                   cg_data.iloc[i]['mg_bound']/Nb,   cg_data.iloc[i]['na_bound']/Na

    ka_kb_combination = [(ln_ka0, ln_kb0) for ln_kb0 in ln_Kb_array for ln_ka0 in ln_Ka_array]
    
    for ln_ka0, ln_kb0 in ka_kb_combination:

        theta_guess  = np.array([0.0, 0.0])

        ionic_strength = 2*cb0 + ca0

        input_data = (ln_ka0, ln_kb0, cb0, ca0, cg_dpgs_conc, ionic_strength, reff)  # I_nacl = ca0

        try:
            theta_solution = fsolve(make_zero_cg, theta_guess, args = input_data)

            if np.all(np.logical_and(theta_solution>=0, theta_solution<=1)):

                error = relative_error(np.array([theta_na, theta_mg]), theta_solution)

                if error <= rel_error_max:
                    rel_error_max = error
                    theta_return = theta_solution
                    lnka0_return = ln_ka0
                    lnkb0_return = ln_kb0
            else:
                continue 

        except FloatingPointError:
            continue

        except Warning:
            continue

    return reff, 100*rel_error_max, np.exp(lnka0_return), np.exp(lnkb0_return), theta_return[0], theta_return[1]


def pickle_and_dump(var, file_name):
    pickle_out = open(file_name, "wb")
    pickle.dump(var, pickle_out)
    pickle_out.close()


cg_data = pd.read_csv("/home/lto/mdanalysis/cg-g2-bound-counterions.csv",\
                      sep = ',', index_col = None,  engine = 'python')
cg_data['theta_na'] = cg_data['na_bound']/24
cg_data['theta_mg'] = cg_data['mg_bound']/12

#print(cg_data)

ln_Ka_array = np.linspace(-10, 10, 2000)
ln_Kb_array = np.linspace(-10, 10, 2000)
reff_array  = np.linspace(1.3, 1.9, 7)
i = 3 
rel_error_max = 0.1
cg_var = make_cg_array() 

for reff in reff_array:

    new_var = np.array(cg_vs_theory(ln_Ka_array, ln_Kb_array, cg_data, reff, i, rel_error_max), dtype = cg_var.dtype)

    cg_var = np.append(cg_var, new_var)

cg_var = np.delete(cg_var, 0)

file_prefix = 'cg_mg_conc_' + str(i) + '_ka_' + str(np.min(ln_Ka_array).astype(int)) + '_' + str(np.max(ln_Ka_array).astype(int)) +\
                             '_kb_'   + str(np.min(ln_Kb_array).astype(int)) + '_' + str(np.max(ln_Kb_array).astype(int)) + '.p'# +\
                             # str(np.round(reff, 1)) + '_error_'+ str(rel_error_max) + '.p'

pickle_and_dump(cg_var, file_prefix)

