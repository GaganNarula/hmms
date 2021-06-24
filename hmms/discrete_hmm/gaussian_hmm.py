from hmms.base.base_hmm import BaseHMM
from scipy.stats import multivariate_normal, invwishart


class GaussianHMM(BaseHMM):
    """Gaussian emission discrete hidden markov model.
        
        Params
        ------
            K : int, number of states
            D : int, number of observation dimensions
            learn_params : str, which parameters to learn
            startprob_prior_conc_weight : float, hyperparamater that controls the concentration
                                            of the dirichlet distribution in each dimension of the 
                                            starting state probability
            transmat_prior_conc_weight : float, same as above, but for each row of the transition 
                                            matrix
            mean_prior : float, prior value for mean (same value used for all dimensions)
            covar_prior_weight : float, prior value for the scale matrix in Inverse Wishart distrib
            estimate_type : str, either ML or MAP for max likelihood of max aposteriori estimate of 
                                params
            do_kmeans : bool, whether to initialze the means and covars with kmeans or gmm
            n_iters : int, number of EM iterations
            tolerance : float, smallest change in loglikelihood below which EM iteration stops
            verbose : bool, whether to print loglikelihood values during EM iterations
            
        
    """
    def __init__(nstates, nfeatures,
                fit_params = 'stmc', 
                init_kmeans = False,
                transmat_dirichlet_conc = 1.,
                startprob_dirichlet_conc = 1.,
                verbose = False,
                n_iters = 50):
        self.nstates = nstates # num hidden states
        self.D = nfeatures # dimensionality of observation, shorthand
        self.nfeatures = nfeatures
        self.n_iters = n_iters # number of EM iterations
        self.do_kmeans = do_kmeans # whether to initialize with kmeans
        # character string e.g. 's' only start probability is updated
        # default = 'stmc' means start, transmat, means and covs
        self.learn_params = learn_params # lea
        self.estimate_type = estimate_type
        self.verbose = verbose
        self.tolerance = tolerance
        if self.estimate_type == 'ML':
            self.startprob_prior_conc = np.ones(self.nstates)
            self.transmat_prior_conc = np.ones(self.nstates)
        else:
            self.transmat_prior_conc = transmat_prior_conc_weight*np.ones(self.nstates)
            self.startprob_prior_conc = startprob_prior_conc_weight*np.ones(self.nstates)
            
        self.prior_mean = mean_prior*np.ones(nfeatures)
        self.covar_prior = covar_prior_weight*np.eye(nfeatures)
        # intialize randomly
        self.transmat = np.random.dirichlet(self.transmat_prior_conc, size = K)
        self.startprob = np.random.dirichlet(self.startprob_prior_conc)
        self.means = multivariate_normal.rvs(size = K, mean = self.prior_mean, cov= np.eye(D))
        self.covs = invwishart.rvs(D+2, self.covar_prior, size = K)
    
    def get_params(self):
        P = {'means':self.means, 
             'covars':self.covs,
             'transmat':self.transmat,
             'startprob':self.startprob}
        return P
    
    def set_params(self, param_dict):
        self.means = param_dict.get('means')
        self.covs = param_dict.get('covs')
        self.transmat = param_dict.get('transmat')
        self.startprob = param_dict.get('startprob')
        
    def _validate_input(self, seqs):
        """seqs is a list of numpy arrays
        """
        assert len(seqs) > 0, 'Empty input array list!'
        assert seqs[0].ndim == 2, 'Need input arrays to be 2 dimensional'
        assert seqs[0].shape[-1] == self.D, 'Mismatch between number of feature dimensions in input and D'
        
    def _init_params(self):
        # intialize randomly
        self.transmat = np.random.dirichlet(self.transmat_prior_conc, size = self.nstates)
        self.startprob = np.random.dirichlet(self.startprob_prior_conc)
        self.means = multivariate_normal.rvs(size = self.nstates, 
                                             mean = self.prior_mean, cov= np.eye(self.D))
        self.covs = invwishart.rvs(self.D + 2, self.covar_prior, size = self.nstates)
        
    def _init_gmm_means_and_covs(self, x, K, covtype = 'diag'):
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components = K, covariance_type = covtype, 
                             random_state = 0, reg_covar = 1e-4)
        gmm.fit(x)
        if covtype == 'spherical':
            # shape (n_components,)
            covs = np.stack([c*np.eye(self.D) for c in gmm.covariances_])
        elif covtype == 'diag':
            # shape (n_components, D)
            covs = np.stack([np.diag(c) for c in gmm.covariances_])
        elif covtype == 'full':
            covs = gmm.covariances_
        return gmm.means_, covs
            
    def _init_stats(self):
        stats = {}
        # collect the statistics for parameter updates
        stats['A'] = np.zeros((self.nstates,self.nstates))
        stats['pi'] = np.zeros(self.nstates)
        stats['mu'] = np.zeros((self.nstates, self.D))
        stats['cov'] = np.zeros((self.nstates, self.D, self.D))
        stats['gammad'] = np.zeros(self.nstates)
        return stats
        
    def log_gaussian_pdf(self, y, z):
        p = multivariate_normal.logpdf(y, mean=self.means[k], cov = self.covars[k])
        return p
    
    def emission_logprobs(self, y):
        ''' Log-likelihood of each step of the observed sequence y for each
            state. P(y | x)
        '''
        self.emission_logprob = np.zeros((y.shape[0], self.nstates))
        for t in range(y.shape[0]):
            for k in range(self.nstates):
                self.emission_logprob[t,k] = self.log_gaussian_pdf(x[t], k)
                
    def posterior_distr(self, y, posterior_type = "filter"):
        ''' Given a sequence y, infer P(x_t | y). 
            Both the smoothed and filtered posterior are computed, 
            one is returned.
            Parameters
            ----------
                y : numpy array, size (timesteps, nfeatures)
                posterior_type : if "filter", only 
            Returns
            ----------
                posterior distribution over all states in sequence.   
        '''
        # first run the forward
        # alpha_hat are filtered posteriors at each step
        alpha_hat, c = forward_recursion_rescaled(y, self.nstates, 
                                                  self.emission_logprob, 
                                                  self.get_params())
        if posterior_type == "filter":
            return alpha_hat
        
        # backward recursion
        beta_hat = backward_recursion_rescaled(y, c, self.nstates, 
                                               self.emission_logprob, 
                                               self.get_params())
        
        # compute smoothed posterior 
        return alpha_hat * beta_hat
    
    def loglikelihood(self, y):
        '''
            Compute loglikelihood of a sequence under the current model
            parameters.
            Parameters
            ----------
                y : array-like. if it is a list, a sequence of length = 
                    len(y) of log likelihood values is returned.
        '''
        if type(y) == list:
            Logl = np.zeros(len(y))
            for i in range(len(y)):
                c = forward_recursion_rescaled(y, self.nstates, 
                                                  self.emission_logprob, 
                                                  self.get_params())
                Logl[i] = np.log(c).sum()
        else:
            # expecting a single 
            c = forward_recursion_rescaled(y, self.nstates, 
                                                  self.emission_logprob, 
                                                  self.get_params())
            Logl = np.log(c).sum()
            
        return Logl
    
    def score(self, seqs):
        """ Compute loglikelihood with forward recursion for a list of sequences. 
        """
        LL =  [self.loglikelihood(x) for x in seqs]
        return np.sum(LL)
    
    def compute_gamma(self, alphahat, betahat):
        """Posterior P(hidden | obs)"""
        return (alphahat * betahat)
    
    def compute_sigma(self, alphahat, betahat, c):
        """Pairwise posterior """
        T = alphahat.shape[0]
        # sigma_matrix has shape [timesteps x prev_states x current_states]
        sigma_matrix = np.zeros((T, self.nstates, self.nstates))
        for t in range(1,T):
            for k in range(self.nstates):
                # k goes over values of latent variable at previous step
                for j in range(self.nstates):
                    # j goes over values of latent at current step
                    sigma_matrix[t,k,j] = (1/c[t]) * alphahat[t-1,k] * np.exp(self.emission_logP[t,j]) \
                                * self.transmat[k,j] * betahat[t,j]
                    
        return sigma_matrix
    
    def E_step(self, x):
        # get emission probabilities
        self.emission_logprobs(x)
        # do forward recursion, get filtered posterior (alphahat) and likelihood terms c
        alphahat, c = forward_recursion_rescaled(x, self.nstates, self.emission_logprob,
                                                self.get_params())
        # do backward recursion
        betahat = backward_recursion_rescaled(x, c, self.nstates, self.emission_logprob,
                                                self.get_params())
        # get posterior
        gamma = self.compute_gamma(alphahat, betahat)
        # get pairwise posterior
        sigma = self.compute_sigma(alphahat, betahat, c)
        
        return alphahat, betahat, gamma, sigma, c

    def _update_mean_stats(self, y, gamma):
        """ mean update statistics """ 
        M = []
        for k in range(self.nstates):
            mu = np.tile(gamma[:,k].reshape(-1,1),(1,self.D)) * y 
            M.append(mu.sum(axis=0))
        return np.stack(M) 
    
    def _update_cov_stats(self, y, gamma):
        ''' covariance update statistics '''
        stats = []
        T = len(y)
        for k in range(self.nstates):
            S = [np.outer(y[t] - self.means_old[k],y[t] - self.means_old[k])*gamma[t,k] for t in range(T)]
            # stack on time axis and sum over time steps
            S = np.stack(S,axis=0).sum(axis=0)
            stats.append(S)
        return np.stack(stats)
    
    def do_Mstep_many_sequences(self, stats):
        ''' M-step for N i.i.d sequences 
            Parameters
            ----------
                stats : dict, contains updates for all parameters
                            fields:
                                - pi startprob
                                - A transmat
                                - mu means
                                - cov covariances
        '''
        # update start probability vector
        if 's' in self.learn_params:
            num = stats['pi'] + self.startprob_prior_conc - 1
            # sum over states
            self.startprob = num / num.sum()
        # update transition matrix, means and covs
        for k in range(self.nstates):
            if 't' in self.learn_params:
                num = stats['A'][k,:] + self.transmat_prior_conc[k] - 1
                self.transmat[k,:] = num / num.sum()
            # for mean
            if 'm' in self.learn_params:
                self.means[k] = (stats['mu'][k] + self.prior_mean)/(stats['gammad'][k] + 1)
            # for covariance
            if 'c' in self.learn_params:
                # prior contribution to numerator
                mu_mu = np.outer(self.means_old[k],self.means_old[k])
                prior = mu_mu + self.covar_prior
                self.covs[k] = (stats['cov'][k] + prior)/(stats['gammad'][k] + (2*self.D + 4))
                                
        
    def fit(self, seqs, log_every = 10):
        """ Fit params to several i.i.d sequences
        """ 
        self._validate_input(seqs)
        
        if self.do_kmeans:
            S = np.concatenate(seqs,axis=0)
            self._init_gmm_means_and_covs(S, self.nstates, 'full')
            
        Lens = [y.shape[0] for y in seqs]
        LLprev = 0.
        for n in range(self.n_iters):
            stats = self._init_stats()
            LL = 0. # log likelihood over all sequences
            for i, y in enumerate(seqs):
                # sigma is [T,nstates,nstates] matrix
                # gamma is [T, nstates] matrix
                _, _, gamma, sigma, c = self.E_step(y)
                LL += np.log(c).sum()
                self.means_old = 1. * self.means
                # accumulate sufficient stats
                stats = self._accumulate_stats(stats, y, gamma, sigma)
            delLL = LL - LLprev
            LLprev = LL*1.
            # do one m-step
            self.do_Mstep_many_sequences(stats)
            if self.verbose and i%log_every == 0:
                print('.....iteration %d, total log LL : %.5f, change : %.5f .....'%(n,LL,delLL))
            
            if n>0 and np.abs(delLL) < self.tolerance:
                print('......stopping early.....')
                break
                
    def _sample_multinomial_index(self, P, n = 1):
        z = np.random.multinomial(n, P)
        return np.where(z == 1)[0][0]
    
    def _sample_Gauss_emission(self, z):
        return multivariate_normal.rvs(size = 1, mean = self.means[z], cov= self.covs[z])
    
    def sample(self, tsteps = 100):
        """sample sequences from HMM """
        states = np.zeros(tsteps, dtype = 'int64')
        emissions = np.zeros((tsteps, self.D))
        for t in range(tsteps):
            if t == 0:
                # start state
                z = self._sample_multinomial_index(self.startprob)
            else:
                # sample from transmat
                z = self._sample_multinomial_index(self.transmat[states[t-1],:])
            states[t] = z
            # sample emission
            emissions[t] = self._sample_Gauss_emission(z)
        return states, emissions
                    
    
        