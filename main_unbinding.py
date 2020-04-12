
#=======================================================================================#
import sys
import os
import numpy as np
import scipy as sp
import matplotlib
#matplotlib.use('Agg') 
from matplotlib import pyplot as plt
from decimal import Decimal
import multiprocessing as mp
from itertools import repeat

import tools_unbinding as tl
import plots as pl
#===================================== Initialize =====================================#
num_cores = mp.cpu_count() ; print('num cores',num_cores)

type_run = 1     # 1 for normal, 2 for parallel

type_val = [1,1,0,0,0]  # [0]-run sim, [1]-correlation plots, [2]-determine production rates 
                        # [3]-final plots (1 to show, 2 to save), [4]-arrhenius plot 

type_dist = 1    # 1 for gaussian, 2 for laplace
if type_dist == 1: type_name = 'gauss'
if type_dist == 2: type_name = 'laplace'

nsim = 50000    # number of catalysts
nconc = 1.0   # initial reactant concentration relative to catalyst
sig = 0.50

temp_val = [50.0] 
tend = [3e09]
x1 = [0.0]  # correlation values 
x2 = [0.0] ; x1_2 = [0.0]
ncorr = np.size(x1)

dist_constants = [1]          # which rate constant to create distribution
non_dist_constants = [0,2]    # rate constants that are not a distribution

dft_ea = [16.0,32.0,32.0]
nrxn = np.size(dft_ea)
sizer = 500

ea = np.zeros((nrxn,nsim))
ea_diff = np.zeros((nrxn,nsim))
ea_corr = np.zeros((nrxn,nsim,ncorr))
nprod = 2 

namer0 = type_name+"_rate_data.txt"
if type_val[2] == 1:
  h = open(namer0,"w") ;  h = open(namer0,"a")
#=======================================================================================#
def temp_run(temps,tend,iii):
    temp = temps + 273.15

    # correlation step
    for ii in range(0,np.size(x1)):

      names = [str(int(temps)),str(ii),str(int(sig)),str(int(dft_ea[2]))]

      nam1 = [type_name, 'kmc', *names,'.txt']
      nam2 = [type_name, 'catalyst', *names,'.txt']

      namer1 = '_'.join([str(v) for v in nam1])
      namer2 = '_'.join([str(v) for v in nam2])

      # initialize information for the catalysts
      catalysts = []
      for i in range(0,nsim):
        catalysts.append(tl.catalyst(nrxn,ea[:,i],ea_corr[:,i,ii]))
        catalysts[i].get_rate_const(ea_corr[:,i,ii],1.0e13,tl.kb_kcal,temp)
        catalysts[i].get_branching_ratio()

      # kmc run
      f = open(namer1,"w") ;  f = open(namer1,"a")
      g = open(namer2,"w") ;  g = open(namer2,"a") 
      tend = tl.kmc_unbinding(catalysts,nconc,[f,g],sizer)
      f.close() ; g.close()
#=======================================================================================#

#=======================================================================================#
def get_prod_rate(namer1):

  kmc_data = np.genfromtxt(namer1, delimiter = ' ')
  sizer = np.shape(kmc_data)[0]
  rate_determine = np.zeros((sizer))
  print(np.shape(rate_determine))
  print(sizer)

  for i in range(sizer-1):
    rate_determine[i] = np.abs(kmc_data[i,1]-kmc_data[i+1,1]) / np.abs(kmc_data[i,0]-kmc_data[i+1,0])*1.0e9

  print(rate_determine)

  fig, ax = plt.subplots()
  titles = ['Decay Fits :     k3 = '+str(dft_ea[2])+'   sig = '+str(sig),
  'Time','Concentration','Dif EQ']
  pl.scatter_plot((kmc_data[:,0]),(rate_determine),titles)
  #pl.scatter_plot((kmc_data[:,0]),kmc_data[:,1],titles)
  plt.savefig('deriv.png')
#=======================================================================================#

#=======================================================================================#
def get_sim_plots(namer1,names,temps):        # makes plots (1 show plots, 2 save plots)

  temp = temps + 273.15

  kmc_data = np.genfromtxt(namer1, delimiter = ' ')
  ma_x, ma_y = tl.mm_integrator(kmc_data[-1,0], 1e4,dft_ea,temp,nconc)

  #print(np.shape(kmc_data))
  #kmc_data = kmc_data[:388,:]

  # kmc vs mass action coverage
  #fig, ax=plt.subplots()
  #titles = ['Concentration vs Time','Time','Concentration','Reactant','Product','Free Sites']
  #pl.scatter_plot(kmc_data[:,0],kmc_data[:,1:],titles)
  #cov_nam = [type_name,'cov',*names,'.png']
  #cov_namer = '_'.join([str(v) for v in cov_nam])
  #plt.savefig(cov_namer)
  #plt.close(fig)

  fig, ax = plt.subplots()

  # compare data to fit functions
  init_k0 = tl.get_rate_const(1.0e13,dft_ea[0],tl.kb_kcal,temp) #/ 1.e2
  init_k1 = tl.get_rate_const(1.0e13,dft_ea[1],tl.kb_kcal,temp)
  init_k2 = tl.get_rate_const(1.0e13,dft_ea[2],tl.kb_kcal,temp)

  print('init vals','%.2e' % init_k0, '%.2e' % init_k1, '%.2e' % init_k2)

  titles = ['Decay Fits :     k3 = '+str(dft_ea[2])+'   sig = '+str(sig),
  'Time','Concentration','Specific']

  init_param = np.array([1.0,init_k1,0.0]) 
  titles[3] = '1 Exp'
  pl.scatter_fit_plot(pl.exp_1,kmc_data[:,0],kmc_data[:,1],init_param,titles,0)

  #init_param = np.array([1.0,1.0,init_k1,init_k2,0.0])
  init_param = np.array([1.0,1.0,init_k1,1e-12,0.0])
  titles[3] = '2 Exp'
  pl.scatter_fit_plot(pl.exp_2,kmc_data[:,0],kmc_data[:,1],init_param,titles,1)

  #init_param = np.array([1.0,1.0,1.0,init_k0,init_k1,init_k2,0.0])
  init_param = np.array([2.0,1.0,1.0,init_k0,1e-09,1e-12,0.0])
  titles[3] = '3 Exp'
  pl.scatter_fit_plot(pl.exp_3,kmc_data[:,0],kmc_data[:,1],init_param,titles,2)


  titles = ['Decay Fits :     k3 = '+str(dft_ea[2])+'   sig = '+str(sig),
  'Time','Concentration','Dif EQ']
  #pl.scatter_plot(ma_x,2.0-ma_y[:,3],titles)
  fit_nam = [type_name,'fit',*names,'.png']
  fit_namer = '_'.join([str(v) for v in fit_nam])
  plt.savefig(fit_namer)     
#=======================================================================================#
def get_turnover_freq(ea_off,y):

  from scipy.signal import savgol_filter

  time_max = 2.0*max(y)
  #time_max = 1.0e13

  # get rate constants for turnover frequency 
  const_off = tl.get_rate_const(1.0e13,ea_off,tl.kb_kcal,273.15+50.0)
  const_on = tl.get_rate_const(1.0e13,dft_ea[0],tl.kb_kcal,273.15+50.0)

  print('log off',np.log10(const_off))

  nbins = 1000  #1000
  nbins2 = 1000  #500
  x = np.linspace(0.0,time_max,nbins2)

  #=============================================================================#
  # fit histogram to non-gaussian pdf
  hist = np.histogram(y,bins=nbins2)
  hist_dist = sp.stats.rv_histogram(hist)
  #laplace_smooth = savgol_filter(hist_dist.pdf(x),31,10)
  #spl = sp.interpolate.UnivariateSpline(x,laplace_smooth,s=0.)
  #x = np.linspace(0.0,time_max,nbins*10)
  #spl3 = sp.interpolate.UnivariateSpline(x,tl.spl2(x,y,spl(x)),s=0.0,k=3)  
  #spl3_integral = sp.interpolate.UnivariateSpline.integral(spl3,0.0,time_max)

  #x = np.linspace(0.0,time_max,nbins2)

  # Log Normal fit
  init = [1.5*sig,np.log(4.4e08)]
  popt, pcov = sp.optimize.curve_fit(pl.log_normal,x,hist_dist.pdf(x),p0=init)
  log_normal_param = popt

  # Weibull fit
  #popt, pcov = sp.optimize.curve_fit(pl.weibull,x,hist_dist.pdf(x),p0=init)
  #print(*popt)
  #modelPredictions = pl.weibull(x, 1e09,1e09)
  #=============================================================================#

  #=============================================================================#
  # Laplace transform functions to integrate
  laplace_fnc = lambda t:  pl.log_normal(t,*popt)*np.exp(-const_off*t) 
  laplace_fnc_spl = lambda t: pl.log_normal(t,*popt)*np.exp(-const_off*t)
  #laplace_fnc_spl = lambda t: spl3(t)/spl3_integral*np.exp(-const_off*t)
  #=============================================================================#

  tin = time_max

  # check pdf fits
  x = np.linspace(0.0,time_max,nbins)
  fig, ax = plt.subplots()
  plt.plot(x,pl.log_normal(x,*log_normal_param),label='Log Normal')
  y_hist = y[(y < time_max)]
  plt.hist((y_hist), bins=nbins2, density=True)
  plt.xlabel('Time (s)')
  plt.title('$f_{cat}(t)$ Log Normal Distribution')
  #plt.legend()
  #plt.xscale('log')
  plt.savefig('pdf_check.png')
  #plt.show()
  plt.close()

  x = np.linspace(0.0,time_max/10,nbins)
  fig, ax = plt.subplots()
  plt.plot(x,laplace_fnc(x),label='Normal')
  plt.plot(x,laplace_fnc_spl(x),label='Spl')
  plt.legend()
  plt.savefig('transform_check.png')
  #plt.show()
  plt.close()

  toff = 5.0e14  # 2.0e12
  if ea_off < 36.0:
    toff = 1.0e13
  if ea_off < 33.0:
    toff = 1.0e12  # 1.0e08
  if ea_off < 32.0:
    toff = 1.0e11
  if ea_off < 30.0:
    toff = 1.0e10
  if ea_off < 28.0:
    toff = 1.0e08
  if ea_off < 25.0:
    toff = 1.0e05
  if ea_off < 23.0:
    toff = 2.0e03


  if ea_off < 25:
    toff = np.inf

  #laplace_tran, _ =sp.integrate.quad(laplace_fnc,1.0e09,np.inf,epsabs=1e-18,limit=5000) # tin
  laplace_tran, _ =sp.integrate.quad(laplace_fnc,0.0,toff,epsabs=1e-18,limit=5000) # tin
  laplace_tran_spl, _ =sp.integrate.quad(laplace_fnc_spl,0.0,tin,limit=500)
  
  print('tran',laplace_tran,laplace_tran_spl,'%.4e' % (laplace_tran-laplace_tran_spl) )

  turnover_freq = laplace_tran /(1/const_on+1/const_off *(1.0 - laplace_tran))
  turnover_freq_spl = laplace_tran_spl /(1/const_on+1/const_off *(1.0 - laplace_tran_spl))

  turnover_freq_fnc = lambda conc: laplace_tran_spl / (1/(const_on*conc) + 1/const_off*(1.0 - laplace_tran_spl))

  print('freq',ea_off,'%.4e' % turnover_freq,'%.4e' % turnover_freq_spl)

  return turnover_freq,turnover_freq_spl
#=======================================================================================#

#=======================================================================================#
def main():

  # Run kMC and mass-action kinetics
  if type_val[0] == 1:

    #====================== Introduce correlation into the system ======================#
    for ii in range(0,ncorr):

      corr_matrix = [[1.0,x1[ii],x2[ii]],
                    [x1[ii],1.0,x1_2[ii]],
                    [x2[ii],x1_2[ii],1.0]]

      # get distriubtion of energies where the DFT energy is the mean
      for i in range(0,nsim):
        for j in dist_constants:  # just want to geta distribution of k_cat right now, not correlated
          if (type_dist == 1): ea[j,i] = np.random.normal(loc=dft_ea[j],scale=sig)    # gaussian distribution
          if (type_dist == 2): ea[j,i] = np.random.laplace(loc=dft_ea[j],scale=sig)   # laplace distribution
          ea_diff[j,i] = ea[j,i] - dft_ea[j]

        for j in non_dist_constants:
          ea[j,i] = dft_ea[j]

      # correlate the energies now based on corr_matrix
      y = tl.get_correlation(corr_matrix,ea_diff)
      for i in dist_constants:
        ea_corr[i,:,ii] = y[i,:] + dft_ea[i]
 
      for i in non_dist_constants:
        ea_corr[i,:,ii] = ea[i,:]

      # plot distribution of individual rate constants
      if type_val[1] == 1:
        names = [str(ii),str(int(sig)),str(int(dft_ea[2]))]
        pl.distribution_plot(ea_corr)
        dist_nam = [type_name,'dist',*names,'.png']
        dist_namer = '_'.join([str(v) for v in dist_nam])
        plt.savefig(dist_namer)

      turnover_dist = np.zeros(np.shape(ea_corr))
      for i in range(nrxn):
        for j in range(nsim):
          turnover_dist[i,j,ii] = 1.0 / tl.get_rate_const(1.0e13,ea_corr[i,j,ii],tl.kb_kcal,273.15+50.0)

      # plot correlated distributions
      if type_val[1] == 2:
        for i in range(0,nrxn):
          for j in range(0,nrxn):
            if (i != j):
              title = [i+1,j+1]
              pl.correlation_plot([ea[i,:],ea[j,:]],[ea_corr[i,:,ii],ea_corr[j,:,ii]],title)
              plt.savefig('correlation_'+str(i)+'_'+str(j)+'.png')


      #for i in range(nrxn):  # fit the turnover data 
      fig, ax = plt.subplots()
      y = ea_corr[1,:,ii]
      plt.hist(y, bins=40, density=True)
      mu, std = sp.stats.norm.fit(y)
      x = np.linspace(min(y),max(y),100) 
      pdf_fitted = sp.stats.norm.pdf(x,mu,std)
      plt.plot((x),pdf_fitted)
      plt.savefig('turnover_dist_'+str(int(sig))+'.png')

      y = (turnover_dist[1,:,ii]) 

      #ea_un = [40.0,38.0,36.0,34.0,33.0,32.0,31.0,30.0,28.0,26.0,25.0,24.0,23.0,22.0,20.0,18.0,17.0]
      ea_un = [32.0]
      turnover_x = np.zeros(np.size(ea_un))
      turnover_rate = np.zeros((np.size(ea_un),2))

      turnover_fil = open('turnover_rate_0'+str(int(10*sig))+'.txt',"w")
      turnover_fil.close()
      turnover_fil = open('turnover_rate_0'+str(int(10*sig))+'.txt',"a")

      for i in range(np.size(ea_un)):
        turnover_rate[i] = get_turnover_freq(ea_un[i],y)
        turnover_x[i] = tl.get_rate_const(1.0e13,ea_un[i],tl.kb_kcal,273.15+50.0)

        dat = np.hstack([turnover_x[i],turnover_rate[i,0]])
        np.savetxt(turnover_fil,dat,newline=" ") ; turnover_fil.write('\n')

      fig, ax = plt.subplots()
      title = ['Unbinding Effects','Unbinding Rate (log 10)','Turnover freq (log 10)','Fit']
      pl.scatter_plot(np.log10(turnover_x),np.log10(turnover_rate[:,0]),title)
      title = ['Unbinding Effects','Unbinding Rate (log 10)','Turnover freq (log 10)','Log Normal']
      #pl.scatter_plot(np.log10(turnover_x),np.log10(turnover_rate[:,0]),title)
      #plt.legend()
      plt.savefig('turnover_0'+str(int(10*sig))+'.png')


      # make figure of all the plots 
      if 1==1:
        sig_val = ['01','08','010','013','015', '018', '020']
        turn_rate = []
        fig, ax = plt.subplots()
        for i in sig_val:
          fil = open('turnover_rate_'+str(i)+'.txt',"r")
          dat = np.genfromtxt(fil)
          plt.plot(np.log10(dat[:,0]),np.log10(dat[:,1]),label=str(int(i)*10/100))
        plt.legend()
        plt.title('Turnover Rate  vs Unbinding Rate')
        plt.xlabel('k$_{off}$ (log10 $s^{-1}$)')
        plt.ylabel('k$_{turnover}$ (log10 $s^{-1}$)')
        #plt.savefig('total_rates.png')
        plt.show()
      
      
      raise SystemExit(0)
    #=======================================================================================#

    #================================ Run over temperatures ================================#
    print('running sim')
    rate_list = []
    if type_run == 1:
      for iii in range(0,np.size(temp_val)):
        job = temp_run(temp_val[iii],tend[iii],iii)
        rate_list.append(job)

    if type_run == 2:
      iterable = zip(temp_val,tend,range(np.size(temp_val)))
      with mp.Pool(num_cores) as p:
          rate_list = p.starmap(temp_run,iterable)
    #=====================================================================================#

  #=======================================================================================#
  if type_val[3] == 1: # make plots without running the sim
    
    for iii in range(0,np.size(temp_val)):
      for ii in range(0,np.size(x1)):

        names = [str(int(temp_val[iii])),str(ii),str(int(sig)),str(int(dft_ea[2]))]
        nam1 = [type_name, 'kmc', *names,'.txt']
        namer1 = '_'.join([str(v) for v in nam1])

        get_sim_plots(namer1,names,temp_val[iii])

        get_prod_rate(namer1)

  #=======================================================================================#
  if type_val[4] == 1:  # arrhenius plot

    if type_val[2] == 1:  # determine production rate from file

      for iii in range(0,np.size(temp_val)):
        for ii in range(0,ncorr):
          
          dat = np.hstack([rate_list[iii][0],rate_list[iii][1][ii],rate_list[iii][2][0,ii],rate_list[iii][2][1,ii]])
          np.savetxt(h,dat, newline=" ") ; h.write('\n')

    if type_val[2] == 1:
      h.close()
    rate_data = np.genfromtxt(namer0, delimiter = ' ')

    temps = rate_data[0::3,0]
    print('temps',temps)
    prod_rate = [rate_data[0::3,2:],rate_data[1::3,2:],rate_data[2::3,2:]]
    print(np.shape(prod_rate))
    pl.arrhenius_plot(temps,prod_rate)
    plt.savefig('arrhenius.png')
  #=======================================================================================#


#================================#
if __name__ == '__main__':
  main()
#================================#
