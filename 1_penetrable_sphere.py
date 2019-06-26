#!/usr/bin/env python
# coding: utf-8

#Poisson-Boltzmann Equation Boundary Value Problem

import scipy.integrate as integrate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import fsolve, minimize
import warnings
import pickle

np.seterr(all = 'raise')
warnings.filterwarnings('error')


################################# Plot parameters #############################
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex = True)
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FormatStrFormatter)
##############################################################################

############################# PB parameters #################################
lb           = 0.78 # SPC/E water at 300K (nm)
pb_prefactor = 4*np.pi*lb
z_d          = -24
reff         = 1.9
std_volume   = 0.6022140857
b2_array     = np.linspace(0, 1, 100)
list_b2      =[(b2_mg, b2_na) for b2_mg in b2_array for b2_na in b2_array]

N             = 500     # grid number for delta rho_s
rmax          = 10
r0            = 1e-4
r             = np.linspace(r0, rmax, N)
initial_y     = np.zeros((2, N), dtype = np.float64)

radius_mg, radius_na, radius_cl = 0.132, 0.129, 0.220
##############################################################################



def ionic_strength(*conc_in_mM):
    mg_conc, na_conc, cl_conc = conc_in_mM
    return 0.5 * ( 4*mg_conc + na_conc + cl_conc ) * .6022 / 1000

def kappa(I_num_per_nm3):
    #I_num_per_nm3 = I_mmolar * std_volume / 1000
    return (8 * np.pi * lb * I_num_per_nm3 )**0.5

def relative_error(x, y):
    return np.linalg.norm(x-y) / np.linalg.norm(x)

def heaviside_s(r, reff, prefactor):
    return prefactor*(1 - np.heaviside(r - reff, 0))

def hard_sphere_dist(r, reff, curvature):
    return 1 - 1/(1 + np.exp(-curvature*(r - reff)))

def vol_excl(r, prefactor, sigma):
    return prefactor * np.exp( -r**2 / (2*sigma**2) )

def packing_frac(r, reff, eta_dpgs, curvature):
    return 1 - eta_dpgs + eta_dpgs/(1 + np.exp(-curvature*(r - reff)))


def rho(r, charge, bulk_conc, phi, b2_fit):
    '''b2_fit is 2*B2*rho_in = \beta \mu_i'''
    return charge * bulk_conc * np.exp( -charge*phi - b2_fit)

'''
def rho_steric(r, charge, volfrac, bulk_conc, phi):

    denominator = 1 - volfrac + volfrac*

    return rho(r, charge, bulk_conc, phi) / denominator
'''

def poisson_boltzmann_b2(r, y):

    y2 = -pb_prefactor*(rho(r, +2, mg_conc, y[0], b2_mg) + rho(r, +1, na_conc, y[0], b2_na) + rho(r, -1, cl_conc, y[0], 0) + 
                        den_s*hard_sphere_dist(r, reff, 40)) - 2*y[1]/r

    return np.vstack((y[1], y2))


# boundary condition residuals
def bc(ya, yb):
    # ya = [y(a), y'(a), y''(a)]
    # yb = [y(b), y'(b), y''(b)]
    #return np.array([ya[1] - e_d , yb[1]]) # Neumann 
    return np.array([ya[1], yb[1]]) # Mixed 


def get_bound_ions(b2_pair, mg_conc, na_conc, r, z_d = -24, reff = 1.9):

    dr               = r[1] - r[0]
    fourpirsq        = 4*np.pi*np.power(r, 2)
    idx_reff         = np.abs(r - reff).argmin()
    idx_int_limit    = np.abs(r - (reff + 0.2)).argmin()
    den_s            = 3*z_d/(4*np.pi*reff**3)
    cl_conc          = 2*mg_conc + na_conc
    b2_mg, b2_na     = b2_pair[0], b2_pair[1]

    res = integrate.solve_bvp(poisson_boltzmann_b2, bc, r, initial_y)
    phi = res.sol(r)[0]

    rho_mg, rho_na = rho(r, +2, mg_conc, phi, b2_mg)/2, rho(r, +1, na_conc, phi, b2_na)

    local_mg, local_na = rho_mg*fourpirsq, rho_na*fourpirsq

    cum_mg = [integrate.trapz(local_mg[0:i], r[0:i], dx = dr) for i in range(idx_int_limit)]
    cum_na = [integrate.trapz(local_na[0:i], r[0:i], dx = dr) for i in range(idx_int_limit)]

    return np.array([cum_mg[idx_reff], cum_na[idx_reff]])


def bound_error(b2_pair, *data):

    mg_conc, na_conc, cg_bound_ions, r = data

    print(mg_conc, na_conc)

    bound_pb = get_bound_ions(b2_pair, mg_conc, na_conc, r)

    return relative_error(cg_bound_ions, bound_pb)


def pickle_and_dump(var, file_name):
    pickle_out = open(file_name, "wb")
    pickle.dump(var, pickle_out)
    pickle_out.close()


############################## Import CG data ###############################

cg_data = pd.read_csv("/home/rohit/google-drive/jupyter-notebooks/cg-g2-bound-counterions.csv", sep = ',', index_col = None,  engine = 'python')
cg_data['theta_na'] = cg_data['na_bound']/24
cg_data['theta_mg'] = cg_data['mg_bound']/12
cg_data['mg_by_na'] = cg_data['mg_bound']/cg_data['na_bound']
cg_data.set_index('conc', inplace = True)


conc = 1
mg_conc = cg_data.loc[conc]['mg_bound']*std_volume/1000
na_conc = 150*std_volume/1000
#mg_bound = cg_data.loc[conc]['mg_bound']
#na_bound = cg_data.loc[conc]['na_bound']

#############################################################################

cg_bound_ions  = cg_data.loc[conc]['mg_bound'], cg_data.loc[conc]['na_bound']

data, b2_guess = (mg_conc, na_conc, cg_bound_ions, r), np.array([0.0, 0.0])
cons           = ({'type': 'ineq', 'fun': lambda x: x[0]}, {'type': 'ineq', 'fun': lambda x: x[1]})
sol            = minimize(bound_error, b2_guess, args = data, constraints = cons, tol = 0.001)
b2_sol         = sol.x
print(b2_sol)
