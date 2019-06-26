#!/usr/bin/env python
# coding: utf-8

#Poisson-Boltzmann Equation Boundary Value Problem

import scipy.integrate as integrate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from itertools import permutations
import pickle

from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex = True)
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FormatStrFormatter)

################################################################################
lb           = 0.78 # SPC/E water at 300K (nm)
four_pi_lb   = 4*np.pi*lb
z_d          = -24
reff         = 1.9
curvature    = 40
den_s        = 3*z_d/(4*np.pi*reff**3)
std_volume   = 0.6022140857

N             = 500
rmax          = 10
r0            = 1e-4
r             = np.linspace(r0, rmax, N)
initial_y     = np.zeros((2, N), dtype = np.float64)
#################################################################################

def ionic_strength(*conc_in_mM):
    mg_conc, na_conc, cl_conc = conc_in_mM
    return 0.5 * ( 4*mg_conc + na_conc + cl_conc ) * std_volume / 1000

def kappa(I_num_per_nm3):
    #I_num_per_nm3 = I_mmolar * std_volume / 1000
    return (8 * np.pi * lb * I_num_per_nm3 )**0.5

def relative_error(x, y):
    return np.linalg.norm(x-y) / np.linalg.norm(x)

def hard_sphere_dist(r):
    return 1 - 1/(1 + np.exp(-curvature*(r - reff)))

def free_vol_frac(r, eta_dpgs):
    return 1 - eta_dpgs*hard_sphere_dist(r)

def rho(r, charge, b2, bulk_conc, phi):
    #b2_fit is 2*B2*rho_in = \beta \mu_i
    return charge * bulk_conc * np.exp( -charge*phi) * free_vol_frac(r, b2)

def poisson_boltzmann(r, y):

    y2 = -four_pi_lb*( rho(r, +2, b2_mg, mg_conc, y[0]) +
                       rho(r, +1, b2_na, na_conc, y[0]) +
                       rho(r, -1, 0    , cl_conc, y[0]) +
                       den_s*hard_sphere_dist(r)) - 2*y[1]/r

    return np.vstack((y[1], y2))

def bc(ya, yb):
    return np.array([ya[1], yb[1]]) # Mixed 


def get_bound_ions(r, number_distribution):
    ''' import scipy.integrate as integrate'''
    dr                = r[1] - r[0]
    idx_reff          = np.abs(r - reff).argmin()
    idx_int_limit     = np.abs(r - (reff + 0.1)).argmin()
    local_number      = number_distribution * 4*np.pi*np.power(r, 2)
    cumulative_number = [integrate.trapz(local_number[0:i], r[0:i], dx = dr) for i in range(idx_int_limit)]

    return cumulative_number[idx_reff]


def pickle_and_dump(var, file_name):
    pickle_out = open(file_name, "wb")
    pickle.dump(var, pickle_out)
    pickle_out.close()

# ## Import  CG data

cg_location    = "cg-g2-bound-counterions.csv" # "/home/khy/google-drive/jupyter-notebooks/cg-g2-bound-counterions.csv"
#jacek_location = "/home/khy/google-drive/jupyter-notebooks/whiskey_jacek.csv"
#cg_location = "/home/rohit/google-drive/jupyter-notebooks/cg-g2-bound-counterions.csv"
#jacek_location = "/home/rohit/google-drive/jupyter-notebooks/whiskey_jacek.csv"

#whiskey_jacek = pd.read_csv(jacek_location, sep = ',', index_col = None,  engine = 'python')
#whiskey_jacek.columns = ['x', 'y']

cg_data = pd.read_csv(cg_location, sep = ',', index_col = None,  engine = 'python')
cg_data['theta_na'] = cg_data['na_bound']/(-z_d)
cg_data['theta_mg'] = cg_data['mg_bound']/(-z_d/2)
cg_data['mg_by_na'] = cg_data['mg_bound']/cg_data['na_bound']
cg_data.set_index('conc', inplace = True)

# ## Find B2

tolmax = 0.24
reff = 1.7
b2_array = np.linspace(0, 1, 100)
b2_array = [(b2_mg, b2_na) for b2_mg in b2_array for b2_na in b2_array]
na_conc  = 150*std_volume/1000


for mg_conc in cg_data.index[1:]:

    results = np.zeros(1, dtype = np.dtype([('b2', np.float64, (2,)), ('theta', np.float64, (2,)), ('error', np.float64)])  )

    cg_bound = np.array([cg_data.loc[mg_conc]['mg_bound'], cg_data.loc[mg_conc]['na_bound']], dtype = np.float64)

    mg_conc  = mg_conc*std_volume/1000
    cl_conc          = 2*mg_conc + na_conc


    for b2_pair in b2_array:

        b2_mg, b2_na = b2_pair

        res = integrate.solve_bvp(poisson_boltzmann, bc, r, initial_y)

        phi = res.sol(r)[0]

        rho_mg  , rho_na   = rho(r, +2, b2_mg, mg_conc, phi)/2, rho(r, +1, b2_na, na_conc, phi)

        bound_mg, bound_na = get_bound_ions(r, rho_mg), get_bound_ions(r, rho_na)

        pb_bound           = np.array([bound_mg, bound_na], dtype = np.float64)

        error              = relative_error(cg_bound, pb_bound)

        if error <= tolmax:

            new_result = np.array([(np.array(b2_pair, dtype=np.float64), pb_bound, error)], dtype = results.dtype)
            results    = np.append(results, new_result)

    results = np.delete(results, 0)

    file_prefix = 'pb_cg_'+ str(int(mg_conc*1000/std_volume)) + '_tol_'   + str(tolmax) + '_reff_1.7.p'

    pickle_and_dump(results, file_prefix)

'''
results = np.zeros(1, dtype = np.dtype([('b2', np.float64, (2,)), ('error', np.float64)])  )

for b2_pair in b2_list:

    b2_mg, b2_na = b2_pair

    res = integrate.solve_bvp(poisson_boltzmann, bc, r, initial_y)

    phi = res.sol(r)[0]

    rho_mg, rho_na = rho(+2, mg_conc, phi, b2_mg)/2, rho(+1, na_conc, phi, b2_na)

    local_mg, local_na = rho_mg*fourpirsq, rho_na*fourpirsq

    cum_mg = [integrate.trapz(local_mg[0:i], r[0:i], dx = dr)
              for i in range(idx_int_limit)]

    cum_na = [integrate.trapz(local_na[0:i], r[0:i], dx = dr)
              for i in range(idx_int_limit)]

    pb_theta = np.array([cum_mg[idx_reff], cum_na[idx_reff]], dtype = np.float64)

    error = relative_error(cg_theta, pb_theta)

    if error < tolmax:
        new_result = np.array([(np.array(b2_pair), error)], dtype = results.dtype)
        results = np.append(results, new_result)

results = np.delete(results, 0)

file_prefix = 'pb_cg_'+ str(mg_conc*1000/std_volume) + '_tol_'   + str(tolmax) + '.p'

pickle_and_dump(results, file_prefix)
'''

'''
whiskey_datapoints = 50
mg_na_bulkratio    = np.linspace(1e-3, 0.61, whiskey_datapoints)
na_conc            = 4*std_volume/1000 # mM*std_volume/1000
mg_conc_array      = na_conc*mg_na_bulkratio
cl_conc_array      = na_conc + 2*mg_conc_array
pairs              = [(mg, na_conc, cl) for mg, cl in zip(mg_conc_array, cl_conc_array)]

for mg_conc, na_conc, cl_conc in pairs:

    #volume_frac = mg_conc*radius_mg**3 + na_conc*radius_na**3 + cl_conc*radius_cl**3

    res = integrate.solve_bvp(poisson_boltzmann, bc, r, initial_y)#, verbose = 2)

    phi = res.sol(r)[0]

    rho_mg, rho_na = rho(+2, mg_conc, phi, 0)/2, rho(+1, na_conc, phi, 0)

    local_mg, local_na = rho_mg*fourpirsq, rho_na*fourpirsq

    cum_mg = [integrate.trapz(local_mg[0:i], r[0:i], dx = dr) for i in range(idx_int_limit)]
    cum_na = [integrate.trapz(local_na[0:i], r[0:i], dx = dr) for i in range(idx_int_limit)]

    bound_mg_array.append(cum_mg[idx_reff])
    bound_na_array.append(cum_na[idx_reff])

bound_mg_array = np.array(bound_mg_array, dtype = np.float64)
bound_na_array = np.array(bound_na_array, dtype = np.float64)

mg_na_boundratio = bound_mg_array / bound_na_array
radius_mg, radius_na, radius_cl = 0.132, 0.129, 0.220
'''


'''
# ## Make whiskey plot

fig = plt.figure()
ax  = fig.add_subplot(111)

plt.plot(mg_na_bulkratio, bound_mg_array,  linewidth = 3, markersize = 10,          label = 'PB theory', color = 'xkcd:blue')

plt.plot(whiskey_jacek['x'], whiskey_jacek['y'], linestyle = '--', marker = 's',         label = 'ITC', color = 'xkcd:red')

plt.grid(True)

ax.tick_params(which = 'major', direction = 'in', length = 8, top = True, right = True)
ax.tick_params(which = 'minor', direction = 'in', length = 4, top = True, right = True)
ax.tick_params(labelsize = 15, grid_alpha = 0.4, grid_linestyle = '--')

ax.set_axisbelow(True)

ax.xaxis.set_major_locator(MultipleLocator(.1))
ax.xaxis.set_minor_locator(AutoMinorLocator(5)) # for the minor ticks,
                                                # use no labels; default NullFormatter

ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

xlabel = r"$\displaystyle C^{\rm bulk}_{\rm Mg^{2+}} / C^{\rm bulk}_{\rm Na^{+}}$"
ylabel = r"$\displaystyle N^{\rm bound}_{\rm Mg^{2+}} / N_{\rm dPGS}$"
#ylabel = r"$\displaystyle N_{\rm Mg^{2+}} / N_{\rm Na^{+}}$"

plt.xlabel(xlabel, fontsize = 20)
plt.ylabel(ylabel, fontsize = 20)

font = {'family': 'serif',
        'color':  'xkcd:black',
        'weight': 'normal',
        'size': 25,
        }

plt.text(0.2, 2.8, "Mg/Na", fontdict=font)

plt.legend(fontsize = 19)#, loc = 0, bbox_to_anchor=(1, 0.5, 0.5, 0.5))
plt.subplots_adjust(top = 1.1)

#plt.savefig('/home/rohit/google-drive/magnesium/mg-netz/images/pb_itc_nmg.png',\
            #bbox_inches = 'tight', dpi = 200)

plt.show()


# In[161]:


cg_boundratio = pd.Series(cg_data['mg_by_na'])
cg_bulkratio  = (cg_data['mg_total']- cg_data['mg_bound'])/                (cg_data['na_total']- cg_data['na_bound'])
md_boundratio = [0.328640, 0.7, 1.058235, 1.424771] # i=1  0.523426
md_bulkratio  = [0.07236672584302155, 0.14405807509309024,
                0.24518605081457234, 0.4533333333333333]

fig = plt.figure()
ax  = fig.add_subplot(111)

plt.plot(cg_bulkratio, cg_boundratio, marker = 'o', linestyle = '--', linewidth = 3, markersize = 10, label = 'CG simulations', color = 'xkcd:green')

plt.plot(md_bulkratio, md_boundratio, marker = 's', linestyle = '--', linewidth = 3, markersize = 10, label = 'MD simulations', color = 'xkcd:light blue')

plt.plot(mg_na_bulkratio, mg_na_boundratio, linestyle = '-', linewidth = 3, markersize = 10, label = 'PB theory', color = 'xkcd:red')

plt.plot(mg_na_bulkratio, mg_na_bulkratio, linestyle = '--', linewidth = 3, markersize = 10, label = 'Reference', color = 'xkcd:black')

plt.grid(True)

ax.tick_params(which = 'major', direction = 'in', length = 8, top = True, right = True)
ax.tick_params(which = 'minor', direction = 'in', length = 4, top = True, right = True)
ax.tick_params(labelsize = 15, grid_alpha = 0.4, grid_linestyle = '--')
ax.set_axisbelow(True)

ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(AutoMinorLocator(5)) # for the minor ticks,
                                                # use no labels; default NullFormatter

ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

xlabel = r"Bulk ratio"
ylabel = r"Bound ratio"
#ylabel = r"$\displaystyle N_{\rm Mg^{2+}} / N_{\rm Na^{+}}$"

plt.xlabel(xlabel, fontsize = 20)
plt.ylabel(ylabel, fontsize = 20)

font = {'family': 'serif',
        'color':  'xkcd:black',
        'weight': 'normal',
        'size': 25,
        }

plt.text(0.2, 2.8, "Mg/Na", fontdict=font)
plt.legend(fontsize = 19, loc = 0, bbox_to_anchor=(1, 0.5, 0.5, 0.5))
plt.subplots_adjust(top = 1.1)

#plt.savefig('/home/rohit/google-drive/magnesium/mg-netz/images/mgna_bound_bulk.png',            bbox_inches = 'tight', dpi = 200)

plt.show()
'''
