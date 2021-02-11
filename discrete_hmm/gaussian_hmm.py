from hmms.base.base_hmm import BaseHMM
from scipy.stats import multivariate_normal
from hmms.base.filter import forward_recursion_rescaled,  backward_recursion_rescaled
from hmms.base.smooth import back


class GaussianHMM(BaseHMM):
    ''' HMM with multivariate Gaussian emissions
    
    '''
    def __init__(nstates, 
                fit_params = 'stmc', 
                init_kmeans = False,
                transmat_dirichlet_prior = 1.,
                startprob_dirichlet_prior = 1.,
                verbose = False,
                n_ters = 50):
        self.nstates = nstates # num hidden states
        self.nfeats = None
        self.verbose = verbose
        self.n_iters = n_iters
        self.fit_params = fit_params
        # gaussian means, one for each state
        self.means = None
        # covariance matrices, one for each state
        self.covars = None
        # transition matrix (nstates , nstates)
        self.transmat = np.random.dirichlet(transmat_dirichlet_prior 
                                            * np.ones(nstates), 
                                            size = nstates)
        # starting state probability vector (nstates, 1)
        self.startprob = np.random.dirichlet(transmat_dirichlet_prior 
                                            * np.ones(nstates), 
                                            size = 1)
    
    def get_params(self):
        P = {'means':self.means, 
             'covars':self.covars,
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
    
    def _sample_multinomial_index(self, P, n = 1):
        z = np.random.multinomial(n, P)
        return np.where(z == 1)[0][0]
    
    def _sample_Gauss_emission(self, k):
        return multivariate_normal.rvs(size = 1, mean = self.means[k], cov= self.covars[k])
    
    def sample(self, tsteps = 100):
        ''' Generate samples from this HMM 
        '''
        states = np.zeros(tsteps, dtype = 'int32')
        emissions = np.zeros((tsteps, self.nfeats))
        for t in range(tsteps):
            if t == 0:
                # start state
                x = self._sample_multinomial_index(self.startprob)
            else:
                # sample from transmat
                x = self._sample_multinomial_index(self.transmat[states[t-1],:])
            states[t] = x
            # sample emission
            emissions[t] = self._sample_Gauss_emission(x)
        return states, emissions
                
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
                P_x_y : posterior distribution over all states.   
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
        P_x_y = alpha_hat * beta_hat
        return P_x_y
    
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
        ''' Take expectation step '''
        alphahat, c = self.forward_recursion_rescaled(y)
        betahat = self.backward_recursion_rescaled(y, c)
        gamma = self.compute_gamma(alphahat, betahat)
        sigma = self.compute_sigma(alphahat, betahat, c)
        return alphahat, betahat, gamma, sigma, c
    
    def M_step(self, y, gamma, sigma):
        # update start prob
        if 's' in self.fit_params:
            self.startprob = gamma[0] / gamma[0].sum()
        # update transition matrix
        if 't' in self.fit_params:
            for k in range(self.nstates):
                for j in range(self.nstates):
                    num = sigma[:,k,j].sum() 
                    denom = sigma[:,k,:].sum()
                    self.transmat[k,j] = num / denom
        # update emission params
        self.update_emission(y, gamma)
        
    def update_emission(self, y, gamma):
        ''' Update emission parameters '''
        T = y.shape[0]
        self.means_old = np.ones_like(self.means)*self.means
        # update means
        if 'm' in self.fit_params:
            for k in range(self.nstates):
                for t in range(T):
                    self.means[k] += gamma[t,k] * y[t]
                denom = gamma[:,k].sum()
                self.means[k] /= denom
        # update covariances
        if 'c' in self.fit_params:
            for k in range(self.nstates):
                S = [np.outer(y[t] - self.means[k],
                              y[t] - self.means[k]) for t in range(T)]
                S = np.stack(S)
                for t in range(T):
                    self.covars[k] += gamma[t,k] * S[t]
                denom = gamma[:,k].sum()
                self.covars[k] /= denom
                    
    def fit(self, y, log_every = 10):
        ''' Fit HMM with EM algorithm 
            Parameters
            ----------
                y : numpy array, (timesteps, nfeatures) single 
                    observation sequence
                log_every : if verbose = True, every log_every iterations
                            print log likelihood
        '''
        self.nfeats = y.shape[1]
        if self.init_kmeans:
            self.init_kmeans_means_and_covs(y)
        else:
            self.init_random_means_and_covs(y)
        # each iteration do M and E steps   
        for n in range(self.n_iters):
            _, _, gamma, sigma, c = self.E_step(y)
            self.M_step(x, gamma, sigma)
            marginal_LL = np.log(c).sum()
            
            if self.verbose and n % log_every == 0:
                print('... Iteration %d , Likelihood = %.5f ...'%(n, marginal_LL))
        
        