
def backward_recursion_rescaled(y, c, K, emission_logprob, params):
    ''' Perform backward filtering given an observed sequence y. 
        Filtering returns terms of the form P(z_t | y_t)
        where z_t is the hidden discrete state at time t
        
        Parameters
        ----------
            y : numpy ndarray, observed sequence, size [timesteps, num_features]
            c : numpy array, size [timesteps,], normalization constants from filtered posterior
            K : int , number of hidden states
            emission_logprobs : function that computes log likelihoods
                                for the sequence y at each time step P(y_t | z_t)
            params : dict, containing parameters for HMM   
            
        Returns
        -------
            betahat : numpy array, estimate of P(z_t | y_{t+1:T}) for all t
                        in the sequence
            logL : float, log - likelihood of the observed sequence y
                                
    '''
    T = x.shape[0] # num time steps
    betahat = np.zeros((T, K))
    beta = np.zeros((T, K))
    betahat[-1,:] = 1.
    beta[-1,:] = 1.
    for t in range(T-2,-1,-1):
        for k in range(K):
            # this is actually beta P(z_t, y_{t+1:T})
            beta[t,k] = np.sum(betahat[t+1,:] * np.exp(emission_logprob[t+1,:]) \
                                  * params['transmat'][k,:])
            betahat[t,k] = beta[t,k] / c[t+1]
    return betahat