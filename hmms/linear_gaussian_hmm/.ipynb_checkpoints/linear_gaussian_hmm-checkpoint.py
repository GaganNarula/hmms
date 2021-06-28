#from hmms.base.base_hmm import BaseHMM
import numpy as np
from kalman_filter_smoother import kalman_filter, kalman_backward, kalman_smoother


class LinearGaussianHMM():
    """Linear Gaussian state space model
        .. math::
            x_{t+1} = Ax_t + Qv_t
            y_t = Cx_t + Rw_t
            v_t ~ \mathbf(N)(0, 1)
            w_t ~ \mathbf(N)(0, 1)
        
        ! Currently works only on one sequence at a time
        
    """
    def __init__(self, nstates, ndims, init='random', stability_guarantee=False,
                do_prior=False):
        
        self.nstates = nstates
        self.ndims = nobsdims
        self.stability_guarantee = stability_guarantee
        self.do_prior = do_prior
        
        # parameter initialization
        if init == 'random':
            self._init_params_random()
        else:
            self._init_params_n4sid()
            
        # useful state variables
        self.filtered_state_mu = None
        self.filtered_state_var = None
        self.smoothed_state_mu = None
        self.smoothed_state_var = None
        self.KalmanGain = None
    
    def _validate_data(self, y):
        assert y.ndim == 2, 'Sequence has to be 2-D , if there is only one feature dimension, reshape with (1,-1)'
        assert y.shape[1] > y.shape[0], 'Sequence length (second dimension, axis=1) needs to be longer than feature dimension axis 0'
        
    def _init_params_n4sid(self):
        """Initialize parameters with N4SID """
        pass
    
    def _init_params_random(self):
        self.A = np.matlib.randn(self.nstates, self.nstates)
        self.C = np.matlib.randn(self.ndims, self.nstates)
        self.Q = np.eye(self.nstates)
        self.R = np.eye(self.ndims)
        self.mu0 = np.random.normal(0., 1., size=self.nstates)
        self.V0 = np.eye(self.nstates)
        
    def get_params(self):
        P = {'A': self.A, 
             'C': self.C, 
             'Q': self.Q, 
             'R':self.R,
            'mu0':self.mu0,
            'V0': self.V0}
        return P
    
    def set_params(self, param_dict):
        self.A = param_dict.get('A')
        self.C = param_dict.get('C')
        self.Q = param_dict.get('Q')
        self.R = param_dict.get('R')
        self.mu0 = param_dict.get('mu0')
        self.V0 = param_dict.get('V0')
        
    def posterior_distr(self, y, posterior_type = 'filter'):
        '''Posterior mean and variance of the latent
            Gaussian state is computed.
        '''
        if posterior_type == 'filter':
            mu, V, _, _, _ = kalman_filter(y, self.get_params())
            return mu, V
        mu, V,_,_ = kalman_smoother(y, self.get_params())
        return mu, V
    
    def loglikelihood(self, y):
        _, _, _, _, c = kalman_filter(y, self.get_params())
        return np.log(c).sum()
    
    def E_step(self, y):
        """E step of EM """
        # forward and backward pass
        params = self.get_params()
        mu, V, P, _, c = kalman_filter(y, params)
        return kalman_backward(y, mu, V, P, params), np.log(c).sum()
        
    def M_step(self, y, muhat, Vhat, paircov_prev, paircov_curr):
        
        # first estimate initial conditions
        self.mu0 = muhat[:,0] # first time step of smoothed posterior mean
        self.V0 = 2*paircov_curr[0] - 2*np.outer(muhat[:,0],muhat[:,0]); # same for covariance

        # with prior
        if self.do_prior:
            self.V0 = (self.V0 / (2*self.nstates + 2)) + np.eye(self.nstates)

        # now all other params
        # transition dynamics
        Anew = np.sum(paircov_prev, axis=0) @ np.linalg.inv(np.sum(paircov_curr[:-1],axis=0))
        
        if self.stability_guarantee:
            # do SVD of Anew 
            e,_ = np.linalg.eig(Anew)
            emax = np.max(np.abs(e))
            if emax > 1:
                Anew = Anew / self.nstates
        

        # observation matrix terms
        cnew_term1 = np.zeros((self.ndims,self.nstates))
        for t in range(T):
            cnew_term1 += np.outer(y[:,t], muhat[:,t].T)
        Cnew = cnew_term1 @ np.linalg.inv(np.sum(paircov_curr,axis=0))
        
        # observation noise covariance
        Rnew = np.zeros((self.ndims, self.ndims))
        for t in range(T):
            Rnew += np.outer(y[:,t], y[:,t]) - Cnew@np.outer(muhat[:,t], y[:,t]) \
                    - np.outer(y[:,t], muhat[:,t])@Cnew.T + Cnew@paircov_curr[t,:,:]@Cnew.T
        
        if self.do_prior:
            Rnew = Rnew / (T + 2*self.ndims + 2);
            # add prior 
            Rnew = Rnew + np.eye(self.ndims)
        else:
            Rnew = Rnew / T
        

        Qnew = np.zeros((self.nstates, self.nstates))
        for t in range(1,T):
            Qnew += paircov_curr[t,:,:] - Anew@paircov_prev[t,:,:].T - paircov_prev[t,:,:]@Anew.T \
                    + Anew@paircov_curr[t-1,:,:]@Anew.T        

        if self.do_prior:
            Qnew = Qnew * (1 / (T-1 + 2*self.nstates + 2))
            # add prior 
            Qnew = Qnew + np.eye(self.nstates)
        else:
            Qnew = Qnew / (1/(T-1))
        
        params = {'A':Anew, 'C': Cnew, 'Q': Qnew, 'R': Rnew,
                 'mu0': mu0, 'V0': V0}
        
        self.set_params(params)

        
    def EM_iteration(self, y):
        # params is a structure of parameters
        # y is data [d x T] d: data dims, T: num steps, p : state dims
        T = y.shape[1]
        
        # Estep , compute expected posterior marginal means and covariances
        muhat, Vhat, paircov_prev, paircov_curr, logL  = self.E_step(y)
        
        self.M_step(muhat, Vhat, paircov_prev, paircov_curr)

        # smoothed obs
        smoothed_obs = params['C'] @ muhat
        return smoothed_obs
    
    def fit(self, seq):
        
        self._validate_data(seq)
        
        for n in range(self.niters):
            smoothed_obs = self.EM_iteration(seq)
            