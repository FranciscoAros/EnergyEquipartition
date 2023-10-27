import numpy as np
import emcee as mc

####################################################################################
### Energy equipartition model
##
#  
#  sigma(m) = a0*(m/m0)^{-eta}
#
#  assuming a scale mass m0 = 1 Msun
#  eta = 0.5 correspond to full equipartition 
#   
#  ** note: the code also allows for eta < 0, althous values close to eta=0 might become negative due stochasticity.

def model_eta(m,a0,eta):
    return a0*(m)**(-eta)


#######################################################################################
### FIT discrete data
###

def eta_dis_lnlike(theta,vpmr,e_vpmr,vpmt,e_vpmt,mass):
    a0, eta = theta
    
    s_mod = model_eta(mass,a0,eta)
    
    aux_01 = -0.5*np.sum(np.log(2*np.pi*(s_mod**2+e_vpmr**2)))
    aux_02 = -0.5*np.sum((vpmr**2/(s_mod**2+e_vpmr**2)))

    aux_03 = -0.5*np.sum(np.log(2*np.pi*(s_mod**2+e_vpmt**2)))
    aux_04 = -0.5*np.sum((vpmt**2/(s_mod**2+e_vpmt**2)))

    return aux_01 + aux_02 + aux_03 + aux_04

def eta_dis_lnprior(theta):
    a0, eta = theta
    
    if 0 < a0  and -0.5 < eta <= 0.5:
        return 0.0
    else:
        return -np.inf

def eta_dis_lnprob(theta,vpmr,e_vpmr,vpmt,e_vpmt,mass):
    lp = eta_dis_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + eta_dis_lnlike(theta,vpmr,e_vpmr,vpmt,e_vpmt,mass)


def fit_dis_eta(vpmr,e_vpmr,vpmt,e_vpmt,mass,init_guess,n_dim=2,n_walkers=100,n_steps=2000,progress=False):
    pos = [init_guess + 1e-4*np.random.randn(n_dim) for i in range(n_walkers)]

    sampler = mc.EnsembleSampler(n_walkers, n_dim, eta_dis_lnprob, args=(vpmr,e_vpmr,vpmt,e_vpmt,mass))
    sampler.run_mcmc(pos, n_steps, progress=progress)

    ###
    samples_chain = sampler.chain
    logLike_chain = sampler.lnprobability

    return samples_chain, logLike_chain


##################################################################################
### MCMC output (extract best fit and error range)

def get_ML_errors(samples,logLike):
    idx_max_like = np.argmax(logLike)
    
    bst = np.mean(samples[idx_max_like])
    
    idx_lw = np.where(samples<bst)[0]
    idx_up = np.where(samples>bst)[0]
    
    ## lower-lim
    if len(idx_lw)>0:
        lwL = bst - np.percentile(samples[idx_lw],32)
    else: 
        lwL = 0
        
    ## upper-lim
    if len(idx_up)>0:
        upL = np.percentile(samples[idx_up],68)-bst
    else: 
        upL = 0
    
    return np.array([bst,lwL,upL])

def get_eta_fit(vpmr,e_vpmr,vpmt,e_vpmt,mass,init_guess,n_burn=800,n_walkers=20,n_steps=1000,progress=False):

    aux_sample_chain, aux_lnLike_chain = fit_dis_eta(vpmr,e_vpmr,vpmt,e_vpmt,mass,init_guess,n_walkers=n_walkers,n_steps=n_steps,progress=progress)
    
    aux_samples = aux_sample_chain[:,n_burn:,:].reshape((-1,2))
    aux_logLike = aux_lnLike_chain[:,n_burn:].reshape(-1)

    aux_ETA = np.zeros(6)
    
    aux_ETA[:3] = get_ML_errors(aux_samples[:,0],aux_logLike)
    aux_ETA[3:] = get_ML_errors(aux_samples[:,1],aux_logLike)
    
    return aux_ETA


########################################################################

