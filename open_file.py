#!/usr/bin/env python

import functions_binding_model as func
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

std_volume    = 0.6022140857        # Standard volume L/mol
N = 34


def ln(a):
    return np.log(a)


def kappa(ca0, xb, cd):
    
    x = [i+2 for i in range(len(xb))]
    vol = np.array([1.43 + i*8e-3 for i in x])
    
    ca0 = ca0*std_volume/1000
    cb0 = np.array([i*cd*1.43 for i in xb])/vol
    ccl = 2*cb0 + ca0
    I   = 2*cb0 + ca0 + ccl
    return np.sqrt(8 * np.pi * .7 * I ), cb0

    

def energy(theta_a, theta_b, xb, reff, ca0, cd, ka0, kb0):

    cd     *= std_volume/1000
    k, cb0  = kappa(ca0, xb, cd)
    t       = np.square(1-theta_a-theta_b)
    elec    = N**2*.7/(2*reff*(1+k*reff))*t
    
    A = -    N*theta_a*(ln(ka0*(ca0 - cd*N*theta_a)/(N*theta_a)) )
    B = -0.5*N*theta_b*(ln(kb0*(cb0 - 0.5*cd*N*theta_b)/(0.5*N*theta_b)) )
    C = A+B
    
    MVH = -N*(1-.5*theta_b)*ln(1-.5*theta_b) + .5*N*theta_b*ln(.5*theta_b)\
           + N*theta_a*ln(theta_a) + N*(1-theta_a-theta_b)*ln(1-theta_a-theta_b)

    T = elec + C + MVH
    
    return {'e':elec, 's': C, 'm': MVH, 't': T}

sol = func.open_and_assign("itc_mg_2_5mM_ka_-5_10_kb_-5_10.p") #"pb_cg_1_tol_0.24_reff_1.7.p")

sol = sol[sol['reff'] == 1.9]

new = np.zeros(1, dtype = sol.dtype)

xp = func.get_exp_data('data.csv', 'mg_2_5mM' )

for item in sol:
    try:
        e = energy(item['theta_a'], item['theta_b'], xp[1], item['reff'], xp[3], xp[4], item['ka0'], item['kb0'])
        if np.abs(e['t'][-1]) < 6:
            new = np.append(new, item)
            #print(5*'\n', item, 2*'\n', e[-1])
    except FloatingPointError:
        continue
print(len(new)) #new['ka0'], new['kb0'])

'''
print(np.average(sol['b2'][198:212, 0]),
      np.average(sol['b2'][198:212, 1]),
      np.average(sol['error'][198:212]) )
'''
