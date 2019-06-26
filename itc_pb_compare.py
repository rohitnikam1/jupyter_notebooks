#!/usr/bin/env python
# coding: utf-8

#import functions_binding_model as f
import scipy.integrate as integrate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex = True)
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FormatStrFormatter)

lb            = 0.78 # SPC/E water at 300K (nm)
four_pi_lb    = 4*np.pi*lb
z_d           = -34
reff          = 2.1
#den_s        = 3*z_d/(4*np.pi*reff**3)
std_volume    = 0.6022140857

N             = 500
rmax          = 10
r0            = 1e-4
r             = np.linspace(r0, rmax, N)
initial_y     = np.zeros((2, N), dtype = np.float64)


def den_s(z_d):
    return 3*z_d/(4*np.pi*reff**3)

def heaviside_s(r, reff, prefactor):
    return prefactor*(1 - np.heaviside(r - reff, 0))

def hard_sphere_dist(r, curvature = 40):
    return 1 - 1/(1 + np.exp(-curvature*(r - reff)))

def free_vol_frac(r, eta_dpgs):
    return 1 - eta_dpgs*hard_sphere_dist(r)

def rho(r, charge, eta_dpgs, bulk_conc, phi):
    return charge * bulk_conc * np.exp( -charge*phi) * free_vol_frac(r, eta_dpgs)

def bc(ya, yb):
    return np.array([ya[1], yb[1]]) # Mixed 


def poisson_boltzmann(r, y):

    y2 = -four_pi_lb*( rho(r, +2, eta_dpgs, mg_conc, y[0]) +
                       rho(r, +1, eta_dpgs, na_conc, y[0]) +
                       rho(r, -1, eta_dpgs, cl_conc, y[0]) +
                       den_s(z_d)*hard_sphere_dist(r) ) - 2*y[1]/r

    return np.vstack((y[1], y2))


def get_bound_ions(r, number_distribution):
    ''' import scipy.integrate as integrate'''

    dr                = r[1] - r[0]
    idx_reff          = np.abs(r - reff).argmin()
    idx_int_limit     = np.abs(r - (reff + 0.1)).argmin()
    local_number      = number_distribution * 4*np.pi*np.power(r, 2)
    cumulative_number = [integrate.trapz(local_number[0:i], r[0:i], dx = dr) for i in range(idx_int_limit)]

    return cumulative_number[idx_reff]



def plotify(x_major_ticks = 0.5, x_minor_ticks = 5, y_major_ticks = 1 , y_minor_ticks = 5):

    plt.grid(True)

    ax.tick_params(which = 'major', direction = 'in',\
               length = 8, top = True, right = True)

    ax.tick_params(which = 'minor', direction = 'in',\
               length = 4, top = True, right = True)

    ax.tick_params(labelsize = 15, grid_alpha = 0.4, grid_linestyle = '--')

    ax.set_axisbelow(True)

    ax.xaxis.set_major_locator(MultipleLocator(x_major_ticks))
    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_ticks)) # for the minor ticks,
                                                # use no labels; default NullFormatter

    ax.yaxis.set_major_locator(MultipleLocator(y_major_ticks))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor_ticks))

    xlabel = r"$\displaystyle C_{{\rm Mg^{2+}},{\rm bulk}}$"
    #xlabel = r"$\displaystyle C^{\rm bulk}_{\rm Mg^{2+}} / C^{\rm bulk}_{\rm Na^{+}}$"
    ylabel = r"$\displaystyle N_{{\rm Mg^{2+}},{\rm bound}}$"
    #ylabel = r"$\displaystyle N_{\rm Mg^{2+}} / N_{\rm Na^{+}}$"

    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)

    '''
    font = {'family': 'serif',
            'color':  'xkcd:black',
            'weight': 'normal',
            'size': 25,
            }

    plt.text(115, 2, r"$\displaystyle C_{{\rm Na^{+}},{\rm bulk}}$",\
             fontdict=font)
    '''
    plt.legend(fontsize = 19, loc = 0, bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
    plt.subplots_adjust(top = 1.1)



vol_init        = 1.43
vol_injection   = 8.00e-3
mg_conc         = 0
mg_conc_tot     = 0
na_conc         = 4*std_volume/1000
cl_conc         = 4*std_volume/1000
dpgs_conc_init  = 0.0638*std_volume/1000
mg_inj          = 15.190*std_volume/1000
eta_dpgs        = 0.32
nmg_inj         = mg_inj * vol_injection
n_dpgs          = dpgs_conc_init * vol_init

jacek_location = "whiskey_jacek.csv"#"/home/rohit/google-drive/jupyter-notebooks/whiskey_jacek.csv"
#"/home/lto/mdanalysis/whiskey_jacek.csv"

whiskey_jacek = pd.read_csv(jacek_location, sep = ',', index_col = None, engine = 'python')
whiskey_jacek.columns = ['x', 'y']
whiskey_jacek.x = whiskey_jacek.x*na_conc*1000/std_volume

injections = list(whiskey_jacek.index + 1)


mg_conc_array       = [mg_conc]
mg_conc_total_array = [mg_conc_tot]
zd_array            = [z_d]

res      = integrate.solve_bvp(poisson_boltzmann, bc, r, initial_y)
phi      = res.sol(r)[0]
rho_na   = rho(r, +1, eta_dpgs, na_conc, phi)

bound_mg_cumulative = 0
bound_mg_array      = [bound_mg_cumulative]
bound_na_cumulative = get_bound_ions(r, rho_na)
bound_na_array      = [bound_na_cumulative]

z_d     += bound_na_cumulative

print(z_d)
zd_array.append(z_d)


for inj in injections:

    vol       = vol_init + inj*vol_injection

    mg_conc_tot += nmg_inj / vol
    mg_conc     += nmg_inj / vol
    cl_conc      = 2*mg_conc + na_conc
    dpgs_conc    = n_dpgs / vol

    res          = integrate.solve_bvp(poisson_boltzmann, bc, r, initial_y)

    phi          = res.sol(r)[0]

    rho_mg       = rho(r, +2, eta_dpgs, mg_conc, phi)/2
    rho_na       = rho(r, +1, eta_dpgs, na_conc, phi)

    bound_mg     = get_bound_ions(r, rho_mg)
    bound_na     = get_bound_ions(r, rho_na)


    if mg_conc < bound_mg*dpgs_conc:

        bound_mg = mg_conc / dpgs_conc
        bound_na = 0

        z_d     += bound_mg
        mg_conc  = 0

    elif z_d + bound_mg + bound_na < 0:

        z_d     += bound_mg + bound_na
        mg_conc -= bound_mg * dpgs_conc

    elif z_d < 0:

        bound_na = 0

        if z_d + bound_mg <= 0:

            z_d     += bound_mg

        else:

            bound_mg = -z_d
            z_d      = 0

        mg_conc -= bound_mg * dpgs_conc

    else:

        bound_mg = 0
        bound_na = 0

    bound_mg_cumulative += bound_mg
    bound_na_cumulative += bound_na

    bound_mg_array.append(bound_mg_cumulative)
    bound_na_array.append(bound_na_cumulative)
    zd_array.append(z_d)
    mg_conc_array.append(mg_conc)
    mg_conc_total_array.append(mg_conc_tot)


bound_mg_array       = np.array(bound_mg_array, dtype = np.float64)
bound_na_array       = np.array(bound_na_array, dtype = np.float64)
mg_conc_array        = np.array(mg_conc_array , dtype = np.float64)*1000/std_volume
mg_conc_total_array  = np.array(mg_conc_total_array , dtype = np.float64)*1000/std_volume
zd_array             = np.array(zd_array      , dtype = np.float64)

datatype       = np.dtype([('bound_mg', np.float64, (len(injections)+1,)),\
                           ('bound_na', np.float64, (len(injections)+1,)),\
                           ('mg_conc' , np.float64, (len(injections)+1,)),\
                           ('z'       , np.float64, (len(injections)+2))])

#result = np.array([(bound_mg_array, bound_na_array, mg_conc_array, zd_array)], dtype = datatype)


fig = plt.figure()
ax  = fig.add_subplot(111)

jet = plt.get_cmap('Accent')

colors = iter(jet(np.linspace(0,1,6)))

#plt.plot(mg_conc_total_array, mg_conc_array, mg_conc_total_array, mg_conc_total_array)

plt.plot(mg_conc_total_array , bound_mg_array, linestyle = '--', linewidth  = 2.5,
         marker = 'o', markersize = 7, label = r"PB $\displaystyle {\rm Mg}^{2+}$")

#plt.plot(mg_conc_total_array , bound_na_array, linestyle = '--', linewidth  = 2.5,
         #marker = '<', markersize = 7, label = r"PB $\displaystyle {\rm Na}^{+}$" )

plt.plot(whiskey_jacek['x'], whiskey_jacek['y'], linestyle = '--', linewidth = 2.5, marker = 's', markersize = 7, label = 'ITC' )

plotify(x_major_ticks = 0.5, x_minor_ticks = 5, y_major_ticks = 1 , y_minor_ticks = 5)

plt.show()

#print(result['bound_mg'], '\n', result['bound_na'], '\n', result['mg_conc'], '\n', result['z'])

#file_prefix = 'pb_laden_itc.p'

#pickle_and_dump(result, file_prefix)
