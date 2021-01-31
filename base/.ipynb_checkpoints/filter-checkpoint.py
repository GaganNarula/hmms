

def forward_recursion_rescaled(y, K, emission_logprobs, params):
    ''' Perform filtering given an observed sequence y. 
        Filtering returns terms of the form P(z_t | y_t)
        where z_t is the hidden discrete state at time t
        
        Params:
        ------
            y : numpy ndarray, observed sequence, size [timesteps, num_features]
            K : int , number of hidden states
            emission_logprobs : function that computes log likelihoods
                                for the sequence y at each time step P(y_t | z_t)
            params : dict, containing parameters for HMM   
            
        Returns:
        -------
            alphahat : numpy array, estimate of P(z_t | y_t) for all t
                        in the sequence
            logL : float, log - likelihood of the observed sequence y
                                
    '''
    # log emission probabilities
    logPe = emission_logprobs(y)
    # num timesteps
    T = y.shape[0]
    # alphahat is the filtering distribution P(z_t | x_t)
    alphahat = np.zeros((T, K))
    c = np.zeros((T, 1)) # normalizations, see Bishop page 628
    # step 1 (python step 0)
    for k in range(K):
        alphahat[0,k] = params['start_prob'][k] * np.exp(logPe[0,k])
    # normalization factors
    c[0] = alphahat[0].sum()
    # posterior distribution
    alphahat[0] = alphahat[0] / c[0]

    # step 2 to T (python step 1 to T-1)
    for t in range(1,T):
        for k in range(K):
            term2 = np.dot(params['transmat'][:,k], alphahat[t-1]) 
            # this is actually alpha, the joint distribution P(x_{1:n}, z_n))
            alphahat[t,k] = term2 * np.exp(logPe[t,k])
        c[t] = alphahat[t].sum()
        alphahat[t] /= c[t]
    # likelihood is product of all 'c' terms
    logL = c.log().sum()
    return alphahat, logL




