import numpy as np
import emcee as mc

####################################################################################
### Mass equipartition models based on Bianchini et al. (2016)
#  
# The model defines a "equipartition mass", stars with masses larger than m_eq are in full equipartition
#
#            \  s_0*exp(-0.5*(m/m_eq))    m < m_eq  
# sigma(m) = \
#            \  s_eq*(m/m_eq)^{-1/2}      m > m_eq 
#
# In this version we use mu = 1/m_eq as fitting parameter for the degree of energy equipartition, therefore:
#
#            \  s_0*exp(-0.5*(m*mu))    m*mu < 1  
# sigma(m) = \
#            \  s_eq*(m*mu)^{-1/2}      m*mu > 1 
#
# The code allows for mu < 0. See Aros & Vesperini (2023) for further details.
#

def model_mu(m,s0,mu):
    aux_s = np.zeros(m.size)
    if mu > 0:
        idx_dn = np.where(m*mu<=1)[0]
        idx_up = np.where(m*mu> 1)[0]
                
        # check 0.5*mu, if 0.5*mu < 1
        if 0.5*mu <= 1:
            aux_s[idx_dn] = s0*np.exp(-0.5*(m[idx_dn]-0.5)*mu)
            aux_s[idx_up] = s0*np.exp(-0.5*(1-0.5*mu))*(m[idx_up]*mu)**(-0.5)
            return aux_s
        
        if 0.5*mu > 1:
            aux_s[idx_dn] = s0*np.exp(-0.5*m[idx_dn]*mu)*np.exp(0.5)*(0.5*mu)**0.5
            aux_s[idx_up] = s0*(m[idx_up]*mu)**(-0.5)*(0.5*mu)**(0.5)
            
            return aux_s
        
    
    elif mu < 0:
        aux_s = s0*np.exp(-0.5*(m-0.5)*mu)
        return aux_s
    
    else:
        aux_s = s0*np.ones(m.size)
        return aux_s

#############################################################################
### FIT  DISCRETE DATA (vel_i,mass_i) 

def mu_dis_lnlike(theta,vpmr,e_vpmr,vpmt,e_vpmt,mass):
    s0, mu = theta
    
    s_mod = model_mu(mass,s0,mu)
    
    aux_01 = -0.5*np.sum(np.log(2*np.pi*(s_mod**2+e_vpmr**2)))
    aux_02 = -0.5*np.sum((vpmr**2/(s_mod**2+e_vpmr**2)))

    aux_03 = -0.5*np.sum(np.log(2*np.pi*(s_mod**2+e_vpmt**2)))
    aux_04 = -0.5*np.sum((vpmt**2/(s_mod**2+e_vpmt**2)))

    return aux_01 + aux_02 + aux_03 + aux_04

def mu_dis_lnprior(theta):
    s0, mu = theta    
    if 0 < s0  and  -100 < mu < 100:
        return 0.0
    else:
        return -np.inf
    
def mu_dis_lnprob(theta,vpmr,e_vpmr,vpmt,e_vpmt,mass):
    lp = mu_dis_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + mu_dis_lnlike(theta,vpmr,e_vpmr,vpmt,e_vpmt,mass)


def fit_dis_mu(vpmr,e_vpmr,vpmt,e_vpmt,mass,init_guess,n_dim=2,n_walkers=100,n_steps=2000,progress=False):
    pos = [init_guess + 1e-4*np.random.randn(n_dim) for i in range(n_walkers)]

    sampler = mc.EnsembleSampler(n_walkers, n_dim, mu_dis_lnprob, args=(vpmr,e_vpmr,vpmt,e_vpmt,mass))
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


def get_mu_fit(vpmr,e_vpmr,vpmt,e_vpmt,mass,init_guess,n_burn=800,n_walkers=20,n_steps=1000,progress=False):
    
    aux_sample_chain, aux_lnLike_chain = fit_dis_mu(vpmr,e_vpmr,vpmt,e_vpmt,mass,init_guess,n_walkers=n_walkers,n_steps=n_steps,progress=progress)
    
    aux_samples = aux_sample_chain[:,n_burn:,:].reshape((-1,2))
    aux_logLike = aux_lnLike_chain[:,n_burn:].reshape(-1)

    aux_MU = np.zeros(6)
    
    aux_MU[:3] = get_ML_errors(aux_samples[:,0],aux_logLike)
    aux_MU[3:] = get_ML_errors(aux_samples[:,1],aux_logLike)
    
    return aux_MU
