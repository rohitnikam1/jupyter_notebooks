#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import MDAnalysis as mda
import multiprocessing 
import pmda.custom

available_cores = multiprocessing.cpu_count()

print(f"available cores is {available_cores}.")
print(f"MDAnalysis version is {mda.__version__}.")



# In[2]:


xtc = "../netz-nacl-150mM-traj/md.xtc"
tpr = "../netz-nacl-150mM-traj/md.tpr"


# In[3]:


u = mda.Universe(tpr, xtc)


# In[4]:


s_atoms  = u.select_atoms('type S')
na_atoms = u.select_atoms('type Na')
hw_atoms = u.select_atoms('type HW_spc')
mg_atoms = u.select_atoms('type Mg')
dpgs_atoms = u.select_atoms('resname PGS')


global reff
global reff_s
reff = 2
reff_s = 0.4

# In[5]:


def norm(a, b):
    vec = np.subtract(a, b)
    return 0.1*np.sqrt(vec.dot(vec))


def qualify_in(pos1, pos2, R):
    if norm(pos1, pos2) <= R:
        return 1
    
    
def find_number_around_sulfates(atomgroup):
    
    #for frame in traj:
        
    com_pos = dpgs_atoms.center_of_mass()
    atoms_positions = atomgroup.positions
        
    atoms_inlist = []
        
    for atom_pos in atoms_positions: 
        if qualify_in(com_pos, atom_pos, reff) == 1:
            atoms_inlist.append(atom_pos)
        
    return len(atoms_inlist)
'''

def s_dist(atom, s_atoms, s_threshold):
    
    for s in s_atoms:
        d = norm(atom.position, s.position)
        
        if d <= s_threshold:
            return 1
            break
'''


# In[11]:


atom_list = pmda.custom.AnalysisFromFunction(find_number_around_sulfates, u, na_atoms)
atom_list.run(n_blocks = 4)
print(atom_list.results)
