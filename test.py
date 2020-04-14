import sys
import os
import numpy as np
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
from decimal import Decimal
from scipy.optimize import curve_fit

prod_time = np.genfromtxt('gauss_prod_time_50_0_0_40_.txt')
prod_time2 = np.genfromtxt('gauss_prod_time_50_0_0_32_.txt')
prod_time3 = np.genfromtxt('gauss_prod_time_50_0_0_30_.txt')

nbins = 60

fig, ax = plt.subplots()
sns.distplot(np.log10(prod_time),hist=False,kde=True,bins=nbins, 
             kde_kws = {'shade': True, 'linewidth': 3},label='40')
sns.distplot(np.log10(prod_time2),hist=False,kde=True,bins=nbins, 
             kde_kws = {'shade': True, 'linewidth': 3},label='32')
sns.distplot(np.log10(prod_time3),hist=False,kde=True,bins=nbins, 
             kde_kws = {'shade': True, 'linewidth': 3},label='32')
plt.legend()
plt.show()
