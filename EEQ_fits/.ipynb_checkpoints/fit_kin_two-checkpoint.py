import numpy as np
import emcee as mc

#######################

def tot_lnlike(theta,vpmr,e_vpmr,vpmt,e_vpmt):
    s0 = theta

    aux_01 = -0.5*np.sum(np.log(2*np.pi*(s0**2+e_vpmr**2)))
    aux_02 = -0.5*np.sum((vpmr**2/(s0**2+e_vpmr**2)))

    aux_03 = -0.5*np.sum(np.log(2*np.pi*(s0**2+e_vpmt**2)))
    aux_04 = -0.5*np.sum((vpmt**2/(s0**2+e_vpmt**2)))

    return aux_01 + aux_02 + aux_03 + aux_04

def tot_lnprior(theta):
    s0 = theta
    
    if 0 < s0:
        return 0.0
    else:
        return -np.inf

def tot_lnprob(theta,vpmr,e_vpmr,vpmt,e_vpmt):
    lp = tot_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + tot_lnlike(theta,vpmr,e_vpmr,vpmt,e_vpmt)


def fit_tot(vpmr,e_vpmr,vpmt,e_vpmt,init_guess,n_dim=1,n_walkers=100,n_steps=1000,progress=False):
    pos = [init_guess + 1e-4*np.random.randn(n_dim) for i in range(n_walkers)]

    sampler = mc.EnsembleSampler(n_walkers, n_dim, tot_lnprob, args=(vpmr,e_vpmr,vpmt,e_vpmt))
    
    sampler.run_mcmc(pos, n_steps, progress=progress)

    ###
    samples_chain = sampler.chain
    logLike_chain = sampler.lnprobability

    return samples_chain, logLike_chain


###################################################################
#####
#####

def get_prf_tot2D_equalnumber(vpmr,e_vpmr,vpmt,e_vpmt,mag,mass,n_bins,n_walkers=100,n_steps=1000,n_burn=750,progress=False):
    aux_EE = np.zeros((5,n_bins))
    idx_bins = np.array_split(np.argsort(mass),n_bins)
    
    for k in range(n_bins):
        idx_sel = idx_bins[k]
        
        aux_EE[0,k] = np.median(mag[idx_sel])
        aux_EE[1,k] = np.median(mass[idx_sel])
        
        #########################################
        init_guess= np.array([6.0])
        
        sample_chain, lnLike_chain = fit_tot(vpmr[idx_sel],e_vpmr[idx_sel],vpmt[idx_sel],e_vpmt[idx_sel],init_guess,n_walkers=n_walkers,n_steps=n_steps,progress=progress)
    
        
        samples = sample_chain[:,n_burn:].reshape(-1)
        logLike = lnLike_chain[:,n_burn:].reshape(-1)
    
        aux_EE[2,k] = np.median(samples)
        aux_EE[3,k] = np.median(samples) - np.percentile(samples,16)
        aux_EE[4,k] = np.percentile(samples,84) - np.median(samples)
        
    return aux_EE
    
def get_prf_tot2D_fixbins(vpmr,e_vpmr,vpmt,e_vpmt,mag,mass,bins,n_walkers=100,n_steps=1000,n_burn=750,progress=False):
    aux_EE = np.zeros((5,bins.size-1))    
    for k in range(bins.size-1):
        idx_sel = np.intersect1d(np.where(mass>bins[k])[0],np.where(mass<=bins[k+1])[0])
        
        aux_EE[0,k] = np.median(mag[idx_sel])
        aux_EE[1,k] = np.median(mass[idx_sel])
        
        #########################################
        init_guess= np.array([6.0])
        
        sample_chain, lnLike_chain = fit_tot(vpmr[idx_sel],e_vpmr[idx_sel],vpmt[idx_sel],e_vpmt[idx_sel],init_guess,n_walkers=n_walkers,n_steps=n_steps,progress=progress)
        
        samples = sample_chain[:,n_burn:].reshape(-1)
        logLike = lnLike_chain[:,n_burn:].reshape(-1)
    
        aux_EE[2,k] = np.median(samples)
        aux_EE[3,k] = np.median(samples) - np.percentile(samples,16)
        aux_EE[4,k] = np.percentile(samples,84) - np.median(samples)
        
    return aux_EE    


#################################################################################################
#
#

def get_kin_radial_prf(R,vpmr,e_vpmr,vpmt,e_vpmt,bins,n_walkers=100,n_steps=1000,n_burn=750):
    
    aux_kin = np.zeros((5,bins.size))
        
    for k in range(bins.size-1):
        idx_sel = np.intersect1d(np.where(R>bins[k])[0],np.where(R<=bins[k+1])[0])
        
        aux_kin[0,k] = np.median(R[idx_sel])
        aux_kin[1,k] = idx_sel.size
        
        #########################################
        init_guess= np.array([6.0])
        
        sample_chain, lnLike_chain = fit_tot(vpmr[idx_sel],e_vpmr[idx_sel],vpmt[idx_sel],e_vpmt[idx_sel],init_guess,n_walkers=n_walkers,n_steps=n_steps)
    
        
        samples = sample_chain[:,n_burn:].reshape(-1)
        logLike = lnLike_chain[:,n_burn:].reshape(-1)
    
        aux_kin[2,k] = np.median(samples)
        aux_kin[3,k] = np.median(samples) - np.percentile(samples,16)
        aux_kin[4,k] = np.percentile(samples,84) - np.median(samples)
        
    return aux_kin

    
