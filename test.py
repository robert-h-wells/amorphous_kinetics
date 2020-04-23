import sys
import os
import numpy as np
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
from decimal import Decimal
from scipy.optimize import curve_fit

prod_time1 = np.genfromtxt('gauss_prod_time_50_0_1_30_.txt')
prod_time1_scaled = np.genfromtxt('gauss_prod_time_50_0_1_30_scaled.txt')

prod_time2 = np.genfromtxt('gauss_prod_time_50_0_1_26_.txt')
prod_time2_scaled = np.genfromtxt('gauss_prod_time_50_0_1_26_scaled.txt')

prod_time3 = np.genfromtxt('gauss_prod_time_50_0_1_24_.txt')
prod_time3_scaled = np.genfromtxt('gauss_prod_time_50_0_1_24_scaled.txt')

prod_time4 = np.genfromtxt('gauss_prod_time_50_0_1_23_5.txt')
prod_time4_scaled = np.genfromtxt('gauss_prod_time_50_0_1_23_5_scaled.txt')


nbins = 60

if 1==1:
    fig, ax = plt.subplots()
    sns.distplot(np.log10(prod_time1[:,2]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='30')
    sns.distplot(np.log10(prod_time2[:,2]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='26')
    sns.distplot(np.log10(prod_time3[:,2]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='24')
    sns.distplot(np.log10(prod_time4[:,2]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='23.5')

    sns.distplot(np.log10(prod_time1_scaled[:,2]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='30 scaled')
    sns.distplot(np.log10(prod_time2_scaled[:,2]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='26 scaled')
    sns.distplot(np.log10(prod_time3_scaled[:,2]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='24 scaled')
    sns.distplot(np.log10(prod_time4_scaled[:,2]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='23.5 scaled')
    plt.title('turnover')
    plt.show()


    fig, ax = plt.subplots()
    sns.distplot(np.log10(prod_time1[:,1]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='30')
    sns.distplot(np.log10(prod_time2[:,1]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='26')
    sns.distplot(np.log10(prod_time3[:,1]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='24')
    sns.distplot(np.log10(prod_time4[:,1]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='23.5')

    sns.distplot(np.log10(prod_time1_scaled[:,1]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='30 scaled')
    sns.distplot(np.log10(prod_time2_scaled[:,1]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='26 scaled')
    sns.distplot(np.log10(prod_time3_scaled[:,1]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='24 scaled')
    sns.distplot(np.log10(prod_time4_scaled[:,1]),hist=False,kde=True,bins=nbins, 
                kde_kws = {'shade': True, 'linewidth': 3},label='23.5 scaled')
    plt.title('tin ')
    plt.show()


fig, ax = plt.subplots()
sns.distplot(np.log10(prod_time4[:,2]),hist=True,kde=True,bins=nbins, 
             kde_kws = {'shade': True, 'linewidth': 3},label='normal')
sns.distplot(np.log10(prod_time4_scaled[:,2]),hist=True,kde=True,bins=nbins, 
             kde_kws = {'shade': True, 'linewidth': 3},label='scaled')
plt.title('turnover vs 4')
plt.show()

fig, ax = plt.subplots()
sns.distplot(np.log10(prod_time4[:,1]),hist=True,kde=True,bins=nbins, 
             kde_kws = {'shade': True, 'linewidth': 3},label='normal')
sns.distplot(np.log10(prod_time4_scaled[:,1]),hist=True,kde=True,bins=nbins, 
             kde_kws = {'shade': True, 'linewidth': 3},label='scaled')
plt.title('tin vs 4')
plt.show()