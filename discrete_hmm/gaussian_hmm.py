from hmms.base.base_hmm import BaseHMM
from scipy.stats import multivariate_normal
from hmms.base.filter import forward_recursion_rescaled,  backward_recursion_rescaled
from hmms.base.smooth import back


class GaussianHMM(BaseHMM):
    ''' HMM with multivariate Gaussian emissions
        Parameters
        ----------
            - nstates : int, number of hidden states
            - init_kmeans : bool, whether to initialize means 
                            and covariances using k-means. Default
                            false. False = initialize with dirichlet
                            prior
            - transmat_
                            
    '''
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
    
    def init_random_means_and_covs(self, y):
        self.means = np.random.normal(loc = 0., scale = 1., 
                                      size = (self.nstates, y.shape[1]))
        self.covars = np.tile(np.eye(self.nfeats), reps=(self.nstates,1,1))
        
    def init_kmeans_means_and_covs(self, y):
        from sklearn.cluster import KMeans
        kms = KMeans(n_clusters = self.nstates)
        kms.fit(y)
        self.means = kms.cluster_centers_
        for k in range(self.nstates):
            self.covars[k] = np.cov(x[kms.labels_ == k], rowvar=False)
        
    def log_gaussian_pdf(self, y, z):
        p = multivariate_normal.logpdf(y, mean=self.means[k], cov = self.covars[k])
        return p
    
    def emission_logprobs(self, y):
        ''' Log-likelihood of each step of the observed sequence y for each
            state. P(y | x)
        '''
        logP = np.zeros((y.shape[0], self.nstates))
        for t in range(y.shape[0]):
            for k in range(self.nstates):
                self.emission_logP[t,k] = self.log_gaussian_pdf(x[t], k)
                
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
    
    def compute_sigma(self, alphahat, betahat, c):
        ''' Term required for posterior weighted expectation 
            of the log transition matrix 
        '''
        T = alphahat.shape[0]
        # sigma_matrix has shape [timesteps x prev_states x current_states]
        sigma_matrix = np.zeros((T, self.nstates, self.nstates))
        for t in range(1,T):
            for k in range(self.nstates):
                # k goes over values of latent variable at previous step
                for j in range(self.nstates):
                    # j goes over values of latent at current step
                    sigma_matrix[t,k,j] = (1/c[t]) * alphahat[t-1,k] 
                                        * np.exp(self.emission_logprobs[t,j]) 
                                        * self.transmat[k,j] * betahat[t,j]
        return sigma_matrix
    
    def E_step(self, y):
        ''' Take expectation step for a sequence y
        '''
        alphahat, c = forward_recursion_rescaled(y, self.nstates, 
                                                  self.emission_logprob, 
                                                  self.get_params())
        betahat = backward_recursion_rescaled(y, c)
        # Posterior P(z_t | y)
        gamma = self.compute_gamma(alphahat, betahat)
        # Posterior pairwise P(z_t ,z_{t-1} | y)
        sigma = self.compute_sigma(alphahat, betahat, c)
        return alphahat, betahat, gamma, sigma, c
    
    def _update_mean_stats(self, y, gamma):
        ''' mean update statistics '''
        M = []
        for k in range(self.nstates):
            mu = np.tile(gamma[:,k].reshape(-1,1),(1,self.D)) \
                    *(y + np.tile(self.prior_mean.reshape(1,-1),(len(y),1)))
            M.append(mu.sum(axis=0))
        return np.stack(M)
    
    def _update_cov_stats(self, y, gamma):
        ''' covariance update statistics '''
        stats = []
        T = len(y)
        for k in range(self.nstates):
            S = [(np.outer(y[t] - self.means[k],y[t] - self.means[k]) \
                 + self.covar_prior)*gamma[t,k] for t in range(T)]
            S = np.stack(S,axis=0).sum(axis=0)
            stats.append(S)
        return np.stack(stats)
        
    def _accumulate_stats(self, stats, y, gamma):
        # sum over sequences, shape [nstates, nstates]
        stats['A'] += np.sum(sigma,axis=0).squeeze() 
        stats['pi'] += gamma[0] # only initial needed
        # for means, multiply each y_t by gamma
        # and add to last sequence
        stats['gammad'] += gamma.sum(axis=0) # for denom
        stats['mu'] += self._update_mean_stats(y, gamma) 
        # for covs
        stats['cov'] += self._update_cov_stats(y, gamma)
        return stats
    
    def do_Mstep_many_sequences(self, stats):
        ''' M-step for N i.i.d sequences 
            Parameters
            ----------
                stats : dict, sufficient statistics for 
                        each parameter update
        '''
        # update start probability vector
        if 's' in self.learn_params:
            self.startprob = self.startprob_prior_conc*stats['pi']
            # sum over states
            self.startprob /= np.sum(self.startprob)
        # update transition matrix, means and covs
        for k in range(self.nstates):
            if 't' in self.learn_params:
                num = self.transmat_prior_conc*stats['A'][k,:] 
                self.transmat[k,:] = num / num.sum()
            # for mean
            if 'm' in self.learn_params:
                self.means[k] = stats['mu'][k]/stats['gammad'][k]
            # for covariance
            if 'c' in self.learn_params:
                self.covs[k] = stats['cov'][k]/(stats['gammad'][k]*(2*self.D + 4))
                                
        
    def fit(self, seqs, log_every = 10):
        ''' Fit HMM with EM algorithm 
            Parameters
            ----------
                seqs : list of numpy arrays, Each array has shape
                        (timesteps, nfeatures) and is a single 
                        observation sequence
                log_every : if self.verbose = True, every log_every
                            iterations print log likelihood
        '''
        if self.do_kmeans:
            self.initKmeans_means_and_covs(np.concatenate(seqs, axis=0))
        LLprev = 0.
        for n in range(self.n_iters):
            stats = self.init_stats()
            LL = 0.
            for i, y in enumerate(seqs):
                # sigma is [T,nstates,nstates] matrix
                # gamma is [T, nstates] matrix
                _, _, gamma, sigma, c = self.E_step(y)
                LL += np.log(c).sum()
                # accumulate sufficient stats
                stats = self._accumulate_stats(self, stats, y, gamma)
            delLL = LL - LLprev
            LLprev = LL*1.
            # do one m-step
            self.do_Mstep_many_sequences(stats)
            if self.verbose and i%log_every == 0:
                print('.....iteration %d, total log LL : %.5f, change : %.5f .....'%(n,LL,delLL))
            
            #if n>0 and delLL < self.tolerance:
            #    print('......stopping early.....')
            #    break
                
    def _sample_multinomial_index(self, P, n = 1):
        z = np.random.multinomial(n, P)
        return np.where(z == 1)[0][0]
    
    def _sample_Gauss_emission(self, z):
        return multivariate_normal.rvs(size = 1, mean = self.means[z], cov= self.covs[z])
    
    def sample(self, tsteps = 100):
        ''' Sample sequences from HMM '''
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
                    
    
        