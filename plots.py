#=============================================================================================================#
import sys
import os
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from decimal import Decimal
from scipy.optimize import curve_fit

import tools_unbinding as tl
#=============================================================================================================#
def hist_plot(data,nbins,title):
  # make histogram plot from data and add a fit of the histogram 
  fig, ax=plt.subplots()

  y, x, _ = plt.hist(data, bins=nbins, density=True, color='b') # density=True
  
  xmin = min(x) - abs(max(x)-min(x)) ; xmax = max(x) + abs(max(x)-min(x))
  x_fit, fit = tl.fit_gaussian(data,xmin,xmax)
  plt.plot(x_fit, fit, 'k')

  #plt.grid(True)
  plt.title(title[0]) 
  plt.xlabel(title[1]) 
  plt.ylabel(title[2])
#=============================================================================================================#

#=============================================================================================================#
def scatter_plot(xdata,ydata,title):
  # simple scatter plot with same xdata but different y values

  #fig, ax=plt.subplots()
  if ydata.ndim > 1:
    for i in range(0,np.shape(ydata)[1]):
      plt.plot(xdata,ydata[:,i],'-',label=title[3+i])
  else:
    plt.plot(xdata,ydata,'-',label=title[3])

  plt.title(title[0])
  plt.xlabel(title[1])
  plt.ylabel(title[2])
  plt.legend(loc=5)
#=============================================================================================================#

def exp_1(t,A,k1,B):
  return A*np.exp(-k1*t)+B

def exp_2(t,A,B,k1,k2,C):
  return A*np.exp(-k1*t)+B*np.exp(-k2*t)+C

def exp_3(t,A,B,C,k1,k2,k3,D):
  return A*np.exp(-k1*t)+B*np.exp(-k2*t)+C*np.exp(-k3*t)+D

def log_normal(t,A,B):
  return 1./(np.sqrt(2.0*np.pi)*A*t)*np.exp(-(np.log(t)-B)**2/(2.0*A**2))

def weibull(t,A,B):
  return A*B*(B*t)**(A-1.0)*np.exp(-(B*t)**A)

#=============================================================================================================#
def scatter_fit_plot(func,xdata,ydata,init,title,val):
  
  popt, pcov = (curve_fit(func, xdata, ydata,p0=init,maxfev = 100400))
  modelPredictions = func((xdata), *popt) 
  fit_val = np.zeros(int(np.size(popt)/2))
  for i in range(0,np.size(fit_val)):
    fit_val[i] = popt[-2-i]

  print(title[3],popt)
  
  plt.plot((xdata),ydata,'.')
  plt.plot((xdata),modelPredictions,'-',label=title[3])
  plt.text(max(xdata)/1.5,2.,'Fit Parameters')
  plt.text(max(xdata)/1.8,1.85-val*0.15,' '.join(['%.2e' % (i,) for i in fit_val]))
  plt.title(title[0])
  plt.xlabel(title[1])
  plt.ylabel(title[2])
  plt.legend(loc=5)
#=============================================================================================================#

#=============================================================================================================#
def scatter_plot_3d(data,data2,data3,title):
  # 3d flat scatter plot
  from scipy.interpolate import griddata

  xi = np.linspace(min(data),max(data),np.size(data))
  yi = np.linspace(min(data2),max(data2),np.size(data2))
  zi = griddata((data,data2), data3, (xi[None,:], yi[:,None]), method='nearest' )

  CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
  CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
  bars = plt.colorbar() # draw colorbar
  bars.set_label(title[3],fontweight='bold')

  plt.xlim(min(data),max(data))
  plt.ylim(min(data2),max(data2))

  plt.title(title[0])
  plt.xlabel(title[1],fontweight='bold',fontsize=12)
  plt.ylabel(title[2],fontweight='bold',fontsize=12)
#=============================================================================================================#
def distribution_plot(data):
  fig, ax = plt.subplots() 

  plt.subplot(3,1,1) ; plt.hist(data[0,:], bins=60, density=True, color='r', label='k_on') ; plt.legend()
  plt.subplot(3,1,2) ; plt.hist(data[1,:], bins=60, density=True, color='b', label='k_cat') ; plt.legend()
  plt.subplot(3,1,3) ; plt.hist(data[2,:], bins=60, density=True, color='g', label='k_off') ; plt.legend() 
  plt.xlabel('Ea (kcal/mol)')
  plt.suptitle('Energetic Barrier Distributions')
#=============================================================================================================#
def correlation_plot(data,data2,title):

  fig, ax = plt.subplots()

  plt.subplot(2,3,1) ; plt.plot(data[0],data[1],'r.') ; plt.title('Original Correlation')
  plt.subplot(2,3,2) ; plt.hist(data[0], bins=60, density=True, color='r') ; plt.title('Rate '+str(title[0]))


  plt.subplot(2,3,3) ; plt.hist(data[1], bins=60, density=True, color='r') ; plt.title('Rate '+str(title[1]))
  plt.subplot(2,3,4) ; plt.plot(data2[0],data2[1],'b.') ; plt.title('Added Correlation')
  plt.subplot(2,3,5) ; plt.hist(data2[0], bins=60, density=True, color='b') ; plt.title('Rate '+str(title[0]))
  plt.subplot(2,3,6) ; plt.hist(data2[1], bins=60, density=True, color='b') ; plt.title('Rate '+str(title[1]))

  plt.suptitle('Added Correlation Result')
#=============================================================================================================#

#=============================================================================================================#
def hist_3d(x,y,z):
  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

  import matplotlib.pyplot as plt
  import numpy as np

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  nbins = 15
  xpos = np.linspace(min(x), max(x), nbins)
  ypos = np.linspace(min(y), max(y), nbins)
  zbin = np.zeros((nbins,nbins))

  for i in range(0,np.size(z)):
    for j in range(0,nbins):
      if (x[i] > xpos[j] and x[i] < xpos[j+1]):
        for k in range(0,nbins):
          if (y[i] > ypos[k] and y[i] < ypos[k+1]):
            zbin[j,k] += z[i]

  ax.bar3d(xpos,ypos,np.zeros(len(zbin)),1,1,zbin)

  #ax.bar3d(x, y, zpos, x_size, y_size, z, zsort='average')
#=============================================================================================================#

#=============================================================================================================#
def hist_3d_multiple(data,title,types):
  from mpl_toolkits.mplot3d import Axes3D

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  nbins = 50

  for i in range(0,np.shape(data)[0]):
  #for i in range(4,8):
    ys = data[i,:] 
    hist, bins = np.histogram(ys, bins=nbins, density=True)
    xs = (bins[:-1] + bins[1:])/2
    z = i+1

    #xmin = bins[0]-bins[-1]+bins[0]/10. ; xmax = bins[-1]
    xmin = 0. ; xmax = bins[-1]
    x_fit, fit = tl.fit_gaussian(data[i,:],xmin,xmax)
    #plt.plot(x_fit,y_fit,fit)

    if types == 1:  # look at weighted gaussian distribution 
      gauss_y_norm = tl.weighted_gaussian([x_fit,fit])
      plt.plot(x_fit,[z]*np.size(x_fit),gauss_y_norm)

    ax.bar(xs, hist, zs=z, zdir='y', alpha=0.8)

  ax.set_title(title[0])
  ax.set_xlabel(title[1])
  #ax.set_ylabel(title[2])
  ax.set_zlabel(title[3])
  ax.set_yticks([]) 

  plt.show()
#=============================================================================================================#

#=============================================================================================================#
def arrhenius_plot(temp_input,rate_input):
  # input temp [K] and rate [s^-1]

  from scipy import stats
   
  temp = 1. / temp_input
  print('rate',rate_input)
  rate = np.log(rate_input)

  ncorr = np.shape(rate_input)[0]
  nprod = np.shape(rate_input)[2]

  print(np.shape(rate))
  print(ncorr,nprod)

  fig, ax=plt.subplots() 
  for i in range(0,ncorr):
    for j in range(0,nprod):
      slope = stats.linregress(temp,rate[i,:,j])[0]
      ea = -slope*tl.kb_kcal
      plt.plot(temp,rate[i,:,j],'-',label=str(i)+' '+str(j)+' '+str(' %.2F' % Decimal(ea)))
      print('ea',' %.2F' % Decimal(ea),i,j)
  plt.legend()
#=============================================================================================================#
