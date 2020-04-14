#=============================================================================================================#
import sys
import os
import random
import numpy as np
import scipy as sp
from scipy.stats import norm
from decimal import Decimal
import matplotlib
from matplotlib import pyplot as plt

import plots as pl

# important constants
gas_const= 8.3144598        # [J/mol*K]
eV = 1.6021766208e-19       # [J/eV]  
pi = 3.1415926536e0         #
hJ = 6.6260695729e-34       # [J*s]
kb = 1.3806488e-23          # [J&K]
kb_ev  = 8.617332478e-5     # [eV&K]
kb_kcal = 0.0019872041
kbhar = 3.166811429e-6      #
kbcm = 0.6950347663e0       # [1&cm*K]
kj_mol_to_eV = 0.010364e0

#=============================================================================================================#
def gaussian(x, mu, sig):
    # create gaussian distribution from mean (mu) and width (sig)
    return 1/(np.sqrt(2*pi*sig**2 )) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#=============================================================================================================#

#=============================================================================================================#
def get_rate_const(A,Ea,kb_type,T):
  # individual rate constant in Arrhenius form
  constant = A * np.exp(-Ea/(kb_type*T) )
  return constant
#=============================================================================================================#

#=============================================================================================================#
def fit_gaussian(data,xmin,xmax):
  # fit data to gaussian distribution
  mu, std = norm.fit(data)
  #print('mu, std',mu,std)
  x_fit = np.linspace(xmin, xmax, 200)
  fit = norm.pdf(x_fit, mu, std)

  return x_fit, fit
#=============================================================================================================#

#=============================================================================================================#
def weighted_gaussian(data):
  # fit data to distribution and create k-weighted distribution, data2
  # data contains x-axis (data[0]) and y-axis (data[1])
  mu, std = norm.fit(data[1])
  mu2 = get_rate_const(1.0e13,mu,kb_kcal,273.15 + 50.)
  fit = data[1]*get_rate_const(1.0e13,data[0],kb_kcal,273.15 + 50.) / mu2

  fit_integrate = sp.integrate.simps(fit,data[0])
  fit_norm = fit / fit_integrate
 
  return fit_norm
#=============================================================================================================#

#=============================================================================================================#
def active_values(x,x_fit,fit,k_fit,temp):
  # determine fraction of sites that give 90% of activity 
  integral = 0.0 ; num = 0 ; active_num = 0.

  while (integral < 0.90):
    integral += (x_fit[num+1]-x_fit[num])*k_fit[num]
    active_num += (x_fit[num+1]-x_fit[num])*fit[num]
    num += 1

  active_ea = x_fit[num] 

  rate = np.zeros((np.size(x)))
  for i in range(0,np.size(x)):
    rate[i] = get_rate_const(1.0e13,x[i],kb_kcal,temp)
  
  effective_rate = np.mean(rate)
  effective_ea = get_ea(1.0e13,effective_rate,kb_kcal,temp)

  return effective_ea, active_ea, active_num
#=============================================================================================================#

#=============================================================================================================#
def get_correlation(corr_matrix,data):
  # introduce correlation onto data from corr_matrix
  from scipy.linalg import cholesky

  sizer = np.shape(corr_matrix)[0]  # 
  c = cholesky(corr_matrix,lower=True)
  y = np.dot(c,data)

  return y
#=============================================================================================================#

#=============================================================================================================#
def eqtns(theta,t,temp):

  rates = np.zeros((3))
  dft_ea = [30.0,32.0,32.0]
  for i in range(0,3):
    rates[i] = get_rate_const(1.0e13,dft_ea[i],kb_kcal,temp)

  dtheta = np.zeros((4))

  dtheta[0] = -rates[0]*theta[0] +(rates[1]+rates[2])*theta[1]
  dtheta[1] = rates[0]*theta[0] - (rates[1]+rates[2])*theta[1]
  dtheta[2] = rates[1]*theta[1]
  dtheta[3] = rates[2]*theta[1]

  return(dtheta)
#=============================================================================================================#

#=============================================================================================================#
def integrator(t_fin,numsteps,temp_val):
  from scipy.integrate import odeint

  t0 = 0.0 ; y0 = np.zeros((4)) ; y0[0] = 1.0
  ts = np.linspace(0, t_fin, numsteps)
  Us = odeint(eqtns, y0, ts, rtol=1e-13, atol=1e-14, full_output=1, mxstep=1000000,args=(temp_val,))
  ys = Us[0]

  final_cov = np.zeros((4))
  final_cov = ys[-1,:]

  return ts,ys
#=============================================================================================================#

#=============================================================================================================#
def mm_difeq(theta,t,ea,temps):
  dtheta = np.zeros((4))  # A,E,AE,P

  k1 = get_rate_const(1.0e13,ea[0],kb_kcal,temps)
  k2 = get_rate_const(1.0e13,ea[1],kb_kcal,temps)
  k3 = get_rate_const(1.0e13,ea[2],kb_kcal,temps)

  dtheta[0] = -k1*theta[0]*theta[1] + k3*theta[2]                # reactant
  dtheta[1] = -k1*theta[0]*theta[1] + k3*theta[2] + k2*theta[2]  # catalyst
  dtheta[2] = k1*theta[0]*theta[1] - k3*theta[2] - k2*theta[2]   # complex 
  dtheta[3] = k2*theta[2]                                        # product

  return(dtheta)
#=============================================================================================================#

#=============================================================================================================#
def mm_integrator(t_fin,numsteps,ea,temp_val,init):
  from scipy.integrate import odeint

  t0 = 0.0 ; y0 = np.zeros((4)) ; y0[0] = init ; y0[1] = 1.
  ts = np.linspace(0, t_fin, numsteps)
  Us = odeint(mm_difeq, y0, ts, rtol=1e-13, atol=1e-14, full_output=1, mxstep=1000000,args=(ea,temp_val,))
  ys = Us[0]

  final_cov = np.zeros((4))
  final_cov = ys[-1,:]

  return ts,ys
#=============================================================================================================#

#=============================================================================================================#
class catalyst:
  # create a catalyst that with nsteps with an energetic barrier for each step

  def __init__(self, nsteps=0, ea=[], ea_corr=[]):
    self.nsteps = nsteps
    self.ea = ea
    self.ea_corr = ea_corr
    self.rate_const = np.zeros((nsteps))
    self.cov = 0.
    self.rate_avail = 0.
    self.production = np.zeros((2))

  def get_rate_const(self,barrier,A,kb_type,T):
    for i in range(0,self.nsteps):
      self.rate_const[i] = A * np.exp(-barrier[i]/(kb_type*T))

  def get_branching_ratio(self):
    self.branch = np.zeros((1,2))
    self.branch[0,0] = self.rate_const[1] / (self.rate_const[1]+self.rate_const[2])
    self.branch[0,1] = self.rate_const[2] / (self.rate_const[1]+self.rate_const[2])

  def get_path_prob(self):
    self.path_prob = np.zeros((2))
    self.path_prob[0] = self.branch[0,0]
    self.path_prob[1] = self.branch[0,1]

  def species_cov(self,species):
    # determine which species is on the catalyst and determine the rates available 
    self.cov = species 
    if self.cov == 0:
      self.rate_avail = 0.0 #self.rate_const[0]  # adsorption reaction (E+S -> ES)
    elif self.cov == 1:
      self.rate_avail = self.rate_const[1]*self.branch[0,0] + self.rate_const[2]*self.branch[0,1] 
      # (ES -> E + P or ES -> E + S)

  def find_action(self,val,rand_val):
    # atom has been selected to perform action so this chooses which action
    if self.cov == 0:
      action_choice = 0     # E+S -> ES
      self.cov = 1 ; self.species_cov(self.cov)
      return(0)
    elif self.cov == 1:
      if val + self.rate_const[1] > rand_val:
        action_choice = 1  # ES -> E + P
        self.cov = 0  ; self.species_cov(self.cov)
        self.production[0] += 1
        return(1)
      elif val + self.rate_const[2] + self.rate_const[1] > rand_val:
        action_choice = 2  # ES -> E + S
        self.cov = 0 ; self.species_cov(self.cov)
        self.production[1] += 1
        return(2)    
#=============================================================================================================#

#=============================================================================================================#
def kmc_run(catalysts,species_conc,fil,sizer):
  # Run kinetic Monte Carlo simulations of an amorphous catalyst system. Michaelis-Menten kinetics.
  #
  # Input catalyst (# of sites and energies). Gas species initial concentration relative to catalyst.
  # Files for data. Sizer limits how much data is stored.

  ncatalyst = np.size(catalysts)
  num_species = int(species_conc*ncatalyst)
  num_species_init = num_species
  conc = [num_species_init,0]
  total_rate = 0.0
  tin = 0.0
  sizer_check = 0
  num_product = 0. 

  # checking desorb process
  ndesorb = 0
  nadsorb = 0
  nrxn = 0

  # Initialize catalysts and find beginning total rate
  for i in range(0,ncatalyst):
    catalysts[i].species_cov(0)
    total_rate += catalysts[i].rate_avail
  
  catalyst_free = ncatalyst
  catalyst_free_list = [i for i in range(ncatalyst)]

  ads_rate_const = catalysts[0].rate_const[0]  # find initial adsorption rate const (all are the same)
  ads_rate = ads_rate_const*catalyst_free*num_species/ncatalyst  # find adsorption rate (k*N_free*[A])
 
  #========== start the simulation ==========#
  while (num_product < num_species_init):  #0.97

    ads_rate = ads_rate_const*catalyst_free*num_species/ncatalyst  # find adsorption rate (k*N_free*[A])

    # Choose time at which next action occurs
    randy = np.random.random()
    delta_t = -1.0/(total_rate+ads_rate)*np.log(randy)
    tin += delta_t 
    randy2 = np.random.random()
    rand_val = randy2*(total_rate+ads_rate)
    
    # Choose atom to have an action
    if rand_val < ads_rate:  # adsorption
      check_val = random.choice(catalyst_free_list) # randomly select from free catalysts
      val = 0
      if catalysts[check_val].rate_avail != 0.0: print('gash')
    else:  # another action

      i = -1 ; check_rate = ads_rate
      while(rand_val > check_rate):  # find which catalyst is responsible for the action
        i += 1
        try:  # sometimes error if rate is very small and randy is ~1
          check_rate += catalysts[i].rate_avail
        except IndexError:
          print('==== i ====',i)
          i -= 1
          break

      check_val = i
      check_rate -= catalysts[check_val].rate_avail
      total_rate -= catalysts[check_val].rate_avail  # remove action being performed from total rate
      val = catalysts[check_val].find_action(check_rate,rand_val) # find which action is occuring

    if val == 0:  # reactant adsorbed
      nadsorb += 1
      catalyst_free -= 1
      catalyst_free_list.remove(check_val)  # remove catalyst from free list
      catalysts[check_val].species_cov(1)   # put molecule on chosen catalyst
      num_species -= 1                      # lose a molecule from "gas" phase
    elif val == 1:  # reactant is consumed
      nrxn += 1
      catalyst_free += 1 
      catalyst_free_list.append(check_val)
      catalysts[check_val].species_cov(0)
      num_product += 1
      conc[0] -= 1
      conc[1] += 1

      # print to file the time of product formation
      dat = np.hstack([tin])
      np.savetxt(fil[2],dat, newline=" ") ; fil[2].write('\n')

    elif val == 2:  # reactant desorbed from catalyst
      ndesorb += 1
      catalyst_free += 1 
      catalyst_free_list.append(check_val)
      catalysts[check_val].species_cov(0)
      num_species += 1
    
    total_rate += catalysts[check_val].rate_avail  # add new action available to total rate

    # limit amount of data written to files
    sizer_check += 1
    if (sizer_check == sizer):
      sizer_check = 0  
      coverage = np.zeros((2))
      for i in range(0,2):
        coverage[i] = float(conc[i])/float(ncatalyst)
      
      free_cov = float(catalyst_free)/float(ncatalyst)
      dat = np.hstack([tin,coverage,free_cov])
      np.savetxt(fil[0],dat, newline=" ") ; fil[0].write('\n')
  
  for i in range(0,ncatalyst):
    dat = np.hstack([i,catalysts[i].ea_corr,catalysts[i].production])
    np.savetxt(fil[1],dat, newline=" ") ; fil[1].write('\n')

  print('nadsorb',nadsorb)
  print('ndesorb',ndesorb)
  print('nrxn',nrxn)
  print('tin',' %.2E' % Decimal(tin))
  return(tin-delta_t)
#=============================================================================================================#
def get_pdf(namer,sig):   
  # create probability density function of the turnover times from kmc run

  turn_dat = np.genfromtxt(namer, delimiter = '\n')
  time_max = 2.0*max(turn_dat)

  nbins = 500
  x = np.linspace(0.10,time_max,nbins)
  hist = np.histogram(turn_dat,bins=nbins)
  hist_dist = sp.stats.rv_histogram(hist)

  init = [1.5*sig,np.log(4.4e08)] 
  popt, pcov = sp.optimize.curve_fit(pl.log_normal,x,hist_dist.pdf(x),p0=init)
  fit_param = popt

  fig, ax = plt.subplots()
  plt.plot(x,pl.log_normal(x,*fit_param))
  plt.hist(turn_dat,bins=200,density=True)
  plt.show()

  moment_fnc = lambda t: t*pl.log_normal(t,*fit_param)
  moment_int, _ =sp.integrate.quad(moment_fnc,0.0,1.0e10,limit=500)
  print('val','%.3e' % (1.0 / moment_int))
#=============================================================================================================#


#=============================================================================================================#
def spl2(x,y,spl):    # used for the full fit spline (not completely sure I stil need it)
  spl2 = np.zeros(np.size(spl))
  for i in range(np.size(x)):
    if spl[i] < 0:
      spl2[i] = 0
    else:
      spl2[i] = spl[i]

    if x[i] < min(y):
      spl2[i] = 0.

    if x[i] > max(y):
      spl2[i] = 0.
  return spl2
#=============================================================================================================#
