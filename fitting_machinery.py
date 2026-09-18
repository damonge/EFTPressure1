#!/usr/bin/env python
# coding: utf-8

# # Fitting Pks Machinery

# ## Setup (from demo)

# In[37]:


import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import matplotlib.pyplot as plt
import pyccl as ccl
import os


# In[38]:


class Pk(object):
    # Class to store power spectrum data
    def __init__(self, kmin, kmax, kavg, pk, nk):
        self.kmin = kmin
        self.kmax = kmax
        self.kavg = kavg
        self.pk = pk
        self.nk = nk


# In[27]:


class SnapPk(object):
    Lbox = 1000.0  # Mpc Flamingo box size
    # Class to organise the power spectrum measurements in a given snapshot
    def __init__(self, namesim, snapnum, kmin=1e-2, kmax=1.0):
        self.sn = snapnum
        self.z = np.loadtxt("FLAMINGO/zs_list.txt")[snapnum]
        self.snapstr = '%04d' % self.sn
        self.sim = namesim
        self.kmin = kmin
        self.kmax = kmax
        self.predir = f'FLAMINGO/L1_m9/{self.sim}/power_spectra'

        # Read and rebin all power spectra
        self.pks = {f'{n1}_{n2}': self.read_pk(self.get_fname_pk(n1, n2))
                    for n1, n2 in self.iter_pairs()}

        # And estimate their Gaussian uncertainties
        self.epks = {f'{n1}_{n2}': self.get_pk_errors(n1, n2)
                     for n1, n2 in self.iter_pairs()}

    def get_pk_covar(self, na, nb, nc, nd):
        # Gaussian Pk covariance
        pkac = self.pks[f'{na}_{nc}'].pk
        pkad = self.pks[f'{na}_{nd}'].pk
        pkbc = self.pks[f'{nb}_{nc}'].pk
        pkbd = self.pks[f'{nb}_{nd}'].pk
        nk = self.pks[f'{na}_{nb}'].nk
        return (pkac*pkbd+pkad*pkbc)/nk

    def get_pk_errors(self, n1, n2):
        # Gaussian error bar
        pk11 = self.pks[f'{n1}_{n1}'].pk
        pk12 = self.pks[f'{n1}_{n2}'].pk
        pk22 = self.pks[f'{n2}_{n2}'].pk
        nk = self.pks[f'{n1}_{n2}'].nk
        return np.sqrt((pk11*pk22+pk12**2)/nk)

    def iter_pairs(self):
        # Returns all pairs of fields (including auto-correlations)
        for n1 in ['matter', 'pressure']:
            for n2 in ['matter', 'pressure']:
                yield n1, n2

    def iter_pairs_unique(self):
        # Returns unique pairs of fields (including auto-correlations)
        names = ['matter', 'pressure']
        for i, n1 in enumerate(names):
            for n2 in names[i:]:
                yield n1, n2

    def get_fname_pk(self, n1, n2):
        # Returns the filename of the power spectrum for a given pair of fields
        if n1 == n2:
            fname1 = fname2 = f'{self.predir}/power_{n1}_{self.snapstr}.txt'
        else:
            fname1 = f'{self.predir}/power_{n1}-{n2}_{self.snapstr}.txt'
            fname2 = f'{self.predir}/power_{n2}-{n1}_{self.snapstr}.txt'

        for fname in [fname1, fname2]:
            if os.path.isfile(fname):
                return fname
        raise KeyError(f'Unknown power spectrum {n1}-{n2}, {fname1}, {fname2}')

    def read_pk(self, fname_pk):
        # Reads a P(k) file in Sara's format and rebins it
        # to a common binning scheme
        d = np.loadtxt(fname_pk, unpack=True)
        ks = d[1]
        pk = d[2]
        # Impose scale cuts
        goodk = (ks >= self.kmin) & (ks <= self.kmax)
        ks = ks[goodk]
        pk = pk[goodk]
        dlk = np.diff(np.log(ks), append=np.log(ks[-1]**2/ks[-2]))
        kmin = np.exp(np.log(ks)-dlk/2)
        kmax = np.exp(np.log(ks)+dlk/2)
        kavg = ks
        nk = (self.Lbox*ks)**3*dlk/(2*np.pi**2)
        return Pk(kmin, kmax, kavg, pk, nk)


# In[29]:


# Create a cosmology object with the same parameters as Flamingo
cosmo = ccl.Cosmology(h=0.681, Omega_b=0.0486, Omega_c=0.306-0.0486, n_s=0.967, sigma8=0.807)


# In[30]:


# Initialise the Eulerian PT calculator.
# Check out the documentation in https://ccl.readthedocs.io/en/latest/api/pyccl.nl_pt.ept.html
# to see the meaning of all parameters.
# We may want to play around with e.g. b1_pk_kind and sub_lowk, for example.
ept = ccl.nl_pt.EulerianPTCalculator(with_NC=True, sub_lowk=False, b1_pk_kind='pt', bk2_pk_kind='linear')
ept.update_ingredients(cosmo)


# In[31]:


# Generate power spectrum interpolators for the different
# EFT templates.
pairs = ['b1:b1', 'b1:b2', 'b1:bs', 'b1:bk2', 'b2:b2', 'b2:bs', 'bs:bs']
pk2d_templates = {pair: ept.get_pk2d_template(pair) for pair in pairs}


# ## Fitting Machinery (from demo)

# In[40]:


from scipy.optimize import curve_fit

def plot_fit(k, pkd, epk, biases, templates, nl = False):
    pkfit = np.dot(templates, biases)
    fig, axes = plt.subplot_mosaic(
        '''
        AAA
        AAA
        AAA
        BBB''', figsize=(6, 6), sharex=True)
    ax = axes['A']
    ax.errorbar(k, pkd, yerr=epk, label='data', fmt='o', markersize=3)
    ax.plot(k, pkfit, label='fit', color='red')

    if nl:
        for b, t, name in zip(biases, templates.T, ['b1:b1', 'b1:b2', 'b1:bs', 'b1:bk2', 'b2:b2', 'b2:bs', 'bs:bs','N']):
            if np.mean(b*t) < 0:
                tplot, ls = -t, '--'
            else:
                tplot, ls = t, '-'
            ax.plot(k, b*tplot, ls, label=name)
    else:    
        for b, t, name in zip(biases, templates.T, ['b1:b1', 'b1:b2', 'b1:bs', 'b1:bk2', 'N']):
                if np.mean(b*t) < 0:
                    tplot, ls = -t, '--'
                else:
                    tplot, ls = t, '-'
                ax.plot(k, b*tplot, ls, label=name)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('P(k) [(Mpc/h)^3]')
    ax.legend()
    ax = axes['B']
    ax.errorbar(k, (pkd-pkfit)/epk, yerr=1, fmt='o', markersize=3)
    ax.axhline(0, color='red', ls='--')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\Delta P(k)/\sigma_P$')
    ax.set_xlabel('k [h/Mpc]')


def fit_pk(snpk, pk_data, pk_error, kmax=0.25, kmin=0.01, verbose=True, plot=True, nl = False):
    # Fit a power spectrum with the EFT templates using curve_fit
    a = 1/(1+snpk.z)

    # Fit a power spectrum with the EFT templates
    k = pk_data.kavg
    pk = pk_data.pk
    epk = pk_error  # Gaussian error bars

    # Select the data to fit
    goodk = (k >= kmin) & (k <= kmax)
    kfit = k[goodk]
    pkfit = pk[goodk]
    epkfit = epk[goodk]

    tk_b1_b1 = pk2d_templates['b1:b1'](kfit, a)
    tk_b1_b2 = pk2d_templates['b1:b2'](kfit, a)
    tk_b1_bs = pk2d_templates['b1:bs'](kfit, a)
    tk_b1_bk2 = pk2d_templates['b1:bk2'](kfit, a)
    tk_b2_b2 = pk2d_templates['b2:b2'](kfit, a)
    tk_b2_bs = pk2d_templates['b2:bs'](kfit, a)
    tk_bs_bs = pk2d_templates['bs:bs'](kfit, a)

    # Define the model function to fit
    def model(k, b1, b2, bs, bk2, N):
        return (b1 * tk_b1_b1 +
                0.5*b2 * tk_b1_b2 +
                0.5*bs * tk_b1_bs +
                0.5*bk2 * tk_b1_bk2 + N)

    def nl_model(k, b1, b2, bs, bk2, N):
        return (b1**2 * tk_b1_b1 +
                b2*b1 * tk_b1_b2 +
                b1*bs * tk_b1_bs +
                0.25 * b2**2 * tk_b2_b2 + 
                0.5 * b2*bs * tk_b2_bs +
                0.25 * bs**2 * tk_bs_bs +
                b1*bk2 * tk_b1_bk2 +
                N)

    # Fit the model to the data using least squares
    if nl:
        popt, pcov = curve_fit(nl_model, kfit, pkfit, sigma=epkfit, maxfev=1000000) #Struggles to converge at default (maxfev = 1200)
        chi2 = np.sum(((pkfit - nl_model(kfit, *popt)) / epkfit) ** 2)
        ndof = len(pkfit) - len(popt)
        if verbose:
            print(f'Fit results: b1={popt[0]:.3e}, b2={popt[1]:.3e}, bs={popt[2]:.3e}, bk2={popt[3]:.3e}, N={popt[4]:.3e}, chi2/ndof={chi2/ndof:.2f}')
            print(popt)
        if plot:
            b1, b2, bs, bk2, N = popt
            amps = np.array([
                b1**2,          
                2*b1*b2,          
                2*b1*bs,          
                b2**2,    
                b2*bs,     
                bs**2,     
                2*b1*bk2,         
                N,              
            ])
            Tmat = np.array([tk_b1_b1, 0.5*tk_b1_b2, 0.5*tk_b1_bs,
                             0.25*tk_b2_b2, 0.5*tk_b2_bs, 0.25*tk_bs_bs,
                             0.5*tk_b1_bk2, np.ones_like(tk_b1_b1)]).T
            plot_fit(kfit, pkfit, epkfit, amps, Tmat, nl = True)
        return popt, pcov, chi2/ndof
    else:
        popt, pcov = curve_fit(model, kfit, pkfit, sigma=epkfit)
        chi2 = np.sum(((pkfit - model(kfit, *popt)) / epkfit) ** 2)
        ndof = len(pkfit) - len(popt)
        if verbose:
            print(f'Fit results: b1={popt[0]:.3e}, b2={popt[1]:.3e}, bs={popt[2]:.3e}, bk2={popt[3]:.3e}, N={popt[4]:.3e}, chi2/ndof={chi2/ndof:.2f}')
        if plot:
            Tmat = np.array([tk_b1_b1, 0.5*tk_b1_b2, 0.5*tk_b1_bs,
                             0.5*tk_b1_bk2, np.ones_like(tk_b1_b1)]).T
            plot_fit(kfit, pkfit, epkfit, popt, Tmat)
        return popt, pcov, chi2/ndof


def fit_pk_analytic(snpk, pk_data, pk_error, kmax=0.25, kmin=0.01, verbose=True, plot=True):
    # Fit a power spectrum with the EFT templates using the analytical least-squares solution
    a = 1/(1+snpk.z)

    # Fit a power spectrum with the EFT templates
    k = pk_data.kavg
    pk = pk_data.pk
    epk = pk_error  # Gaussian error bars

    # Select the data to fit
    goodk = (k >= kmin) & (k <= kmax)
    kfit = k[goodk]
    pkfit = pk[goodk]
    epkfit = epk[goodk]

    tk_b1_b1 = pk2d_templates['b1:b1'](kfit, a)
    tk_b1_b2 = pk2d_templates['b1:b2'](kfit, a)
    tk_b1_bs = pk2d_templates['b1:bs'](kfit, a)
    tk_b1_bk2 = pk2d_templates['b1:bk2'](kfit, a)



    tk_N = np.ones_like(tk_b1_b1)
    Tmat = np.array([tk_b1_b1, 0.5*tk_b1_b2, 0.5*tk_b1_bs, 0.5*tk_b1_bk2, tk_N]).T
    Cov = np.diag(epkfit**2)
    iCov = np.linalg.inv(Cov)
    TiCT = np.dot(Tmat.T, np.dot(iCov, Tmat))
    TiCp = np.dot(Tmat.T, np.dot(iCov, pkfit))
    pcov = np.linalg.inv(TiCT)
    popt = np.dot(pcov, TiCp)
    chi2 = np.sum(((pkfit - np.dot(Tmat, popt)) / epkfit) ** 2)
    ndof = len(pkfit) - len(popt)
    if verbose:
        print(f'Fit results: b1={popt[0]:.3e}, b2={popt[1]:.3e}, bs={popt[2]:.3e}, bk2={popt[3]:.3}, N={popt[4]:.3e}, chi2/ndof={chi2/ndof:.2f}')
    if plot:
        plot_fit(kfit, pkfit, epkfit, popt, Tmat)
    return popt, pcov, chi2/ndof


# In[ ]:


def coefficient_plotter(snpk, kmaxes, which = None):

    names = ['b1', 'b2', 'bs', 'bk2', 'N']

    if which is None:
        which = names

    for idx, name in enumerate(names):

        if name in which:
            parameters, errors = [], []
            parameters_nl, errors_nl = [], []

            for km in kmaxes:

                popt, pcov, _ = fit_pk(snpk, snpk.pks['matter_pressure'], snpk.epks['matter_pressure'], kmax=km, verbose=False, plot=False)
                popt_nl,  pcov_nl, _ = fit_pk(snpk, snpk.pks['pressure_pressure'], snpk.epks['pressure_pressure'], kmax=km, verbose=False, plot=False, nl = True)
                parameters.append(popt[idx])
                errors.append(np.sqrt(np.diag(pcov))[idx])
                parameters_nl.append(popt_nl[idx])
                errors_nl.append(np.sqrt(np.diag(pcov_nl))[idx])

            fig, ax = plt.subplots()
            ax.errorbar(kmaxes, parameters, yerr = errors, label = 'matter-pressure',
                        fmt='o-',        
                        capsize=3,     
                        capthick=1,     
                        elinewidth=1,   
                        markersize=4,
                       alpha = 0.7)
            ax.errorbar(kmaxes, parameters_nl, yerr = errors_nl, label = 'pressure-pressure',
                        fmt='o-',        
                        capsize=3,     
                        capthick=1,     
                        elinewidth=1,   
                        markersize=4,
                        alpha = 0.7)

            ax.set_xlabel(r'$k_{\max}$ [h/Mpc]')
            ax.set_ylabel('Bias value')
            ax.legend()
            ax.set_title(f'{name} against k');

