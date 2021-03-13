from hmms.base.base_hmm import BaseHMM

class LinearGaussianHMM(BaseHMM):
    
    def __init__(self):
        
    def get_params(self):
        P = {'A': self.A, 
             'C': self.C, 
             'Q': self.Q, 
             'R':self.R}
        return P
    
    def posterior_distr(self, y, posterior_type = 'filter'):
        ''' 
            Posterior mean and variance of the latent
            Gaussian state is computed.
        '''
        if posterior_type == 'filter':
            mu, V, _, _, _ = kalman_filter(y, self.get_params())
        return kalman_smoother(x, self.get_params())
    
    def loglikelihood(self, y):
        _, _, _, _, c = kalman_filter(y, self.get_params())
        return np.log(c).sum()
    
    def fit(self, y):
        # params is a structure full of parameters
        # x is data [d x T] d: data dims, T: num steps, p : state dims
        T = x.shape[1]
        d = x.shape[0] # num dimensions
        p = self.

        # Estep , compute expected posterior marginal means and covariances

        # forward pass
        self.posterior_distr(y)
        
        state_out.filtered_state_mu = mu;
        state_out.filtered_state_var = V;
        state_out.smoothed_state_mu = muhat;
        state_out.smoothed_state_var = Vhat;
        state_out.KalmanGain = K;


        # Mstep 

        # first estimate initial conditions
        params.mu0 = muhat[:,1] # first time step of smoothed posterior mean
        params.V0 = 2*squeeze(paircov_curr(1,:,:)) - 2*muhat(:,1)*muhat(:,1)'; % same for covariance

        # with prior
        if do_prior
            params.V0 = (params.V0 / (2*p + 2)) + eye(p,p);
        end

        # now all other params
        # transition dynamics
        Anew = squeeze(sum(paircov_prev,1)) / squeeze(sum(paircov_curr(1:end-1,:,:),1));


        if stability_guarantee:
            # do SVD of Anew 
            e = eig(Anew)
            emax = max(abs(e))
            if emax > 1:
                Anew = Anew / p;
        end

        # observation matrix terms
        cnew_term1 = zeros(d,p);

        for t = 1:T
            cnew_term1 = cnew_term1 + x(:,t)*muhat(:,t)';
        end
        Cnew = cnew_term1 / squeeze(sum(paircov_curr,1));

        Rnew = zeros(d,d);
        for t = 1:T
            Rnew = Rnew + x(:,t)*x(:,t)' - Cnew*(muhat(:,t)*x(:,t)') - ...
                (x(:,t)*muhat(:,t)')*Cnew' + Cnew*squeeze(paircov_curr(t,:,:))*Cnew';
        end
        if do_prior
            Rnew = Rnew / (T + 2*d + 2);
            % add prior 
            Rnew = Rnew + eye(d,d);
        else
            Rnew = Rnew / T;
        end

        #Rnew = (Rnew.' + Rnew)/2;
        #f any(diag(Rnew) <= 0.)
        #    Rnew = Rnew + 1e-3*eye(p,p);
        

        Qnew = zeros(p,p);
        for t = 2:T
            Qnew = Qnew + squeeze(paircov_curr(t,:,:))  ...
                - Anew*squeeze(paircov_prev(t,:,:))' -  ...
                squeeze(paircov_prev(t,:,:))*Anew' +  ...
                Anew*squeeze(paircov_curr(t-1,:,:))*Anew';
        

        if do_prior
            Qnew = Qnew * (1 / (T-1 + 2*p + 2));
            # add prior 
            Qnew = Qnew + eye(p,p);
        else
            Qnew = Qnew / (1/(T-1));
        

        #get symmetric Q back
        #Qnew = (Qnew.' + Qnew)/2;
        #if any(diag(Rnew) <= 0.)
        #    Qnew = Qnew + 1e-3*eye(p,p);

        params['A'] = Anew;
        params['C'] = Cnew;
        params['Q'] = Qnew; 
        params['R'] = Rnew; 


        # smoothed obs
        state_out.smoothed_obs = params.C * muhat;