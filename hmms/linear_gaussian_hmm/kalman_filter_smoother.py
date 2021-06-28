import numpy as np
from scipy.stats import multivariate_normal as mvn
import pdb



def kalman_filter(x, params):
    ''' Kalman filtering algorithm
        Parameters
        ----------
            x : numpy array [d x T], observation sequence
            params : parameters of the model
        Returns
        -------
            mu : filtered state estimate
            V : filtered state covariance
            P : utility matrix
            K : Kalman gain
            c : log likelihood of x under this model
    '''
    # x and mu are both [d x T] and [p x T] matrices
    A = params['A'] # transition dynamics
    C = params['C'] # observation matrix
    Q = params['Q'] # state noise
    R = params['R'] # obs noise 
    mu0 = params['mu0'] # prior mean
    V0 = params['V0'] # prior variance
    
    d = x.shape[0] # obs dim
    p = A.shape[0] # state dim
    T = x.shape[1] # num time steps
    
    mu = np.zeros((p,T)) #% state mean
    P = np.zeros((T,p,p)) #% state covariance
    V = np.zeros((T,p,p)) #% state covariance
    K = np.zeros((T,p,d)) #% Kalman zgain
    c = np.zeros((T,1)) #% normalization constants
    for t in range(T):
        if t == 1:
            K[t] = V0 @ (C.T @ np.linalg.inv(C @ (V0 @ C.T) + R));  
            mu[:,t] = mu0 + K[t]@(x[:,t] - (C@mu0));
            V[t] = (np.eye(p) - (K[t]@C))@V0 ;
            P[t] = A@(V[t]@A.T) + Q;
            try:
                c[t] = mvn.pdf(x[:,t], C@mu0, C@(V0@C.T) + R);
            except:
                break
            
        else:
            # from David Barber Book
            P[t] = A@V[t-1]@A.T + Q
            K[t] = P[t] @ (C.T @ np.linalg.inv(C@P[t]@C.T + R))
            mu[:,t] = A@mu[:,t-1] + K[t]@(x[:,t] - C@(A@mu[:,t-1]))
            # joseph's symmetrized update for covariance
            Term1 = np.eye(p) - K[t]@C 
            Term2 = K[t]@(R@K[t].T)
            V[t] = Term1@P[t]@Term1.T + Term2
            cv = C@P[t]@C.T + R
            try:
                c[t] = mvn.pdf(x[:,t], C@(A@mu[:,t-1]), cv)
            except:
                pdb.set_trace()
                
    return mu, V, P, K, c


def kalman_backward(x, mu, V, P, params):
    '''
        Kalman backward recursion.
        Parameters
        ----------
            x : observation sequence
            mu : filtered posterior mean
            V : filtered posterior covariance
            P : util matrix
            params : parameters
        Returns
        -------
            muhat : smoothed posterior mean
            Vhat : smoothed posterior covariance
            paircov_prev : pairwise posterior 
            paircov_curr : pairwise posterior current
    '''
    A = params['A']
    p = A.shape[1] #state dim
    T = x.shape[1] #num time steps
    muhat = np.zeros((p,T)) #state mean
    Vhat = np.zeros((T,p,p)) #state covariance
    J = np.zeros((T,p,p)) # useful matrix for learning
    for t in np.arange(T,1,-1):
        if t == T:
            muhat[:,t] = mu[:,t]
            Vhat[t] = V[t]
        else:
            J[t] = V[t] @ (A.T @ np.linalg.inv(P[t]))
            
            muhat[:,t] = mu[:,t] + J[t]@(muhat[:,t+1] - A@mu[:,t])
            
            Vhat[t] = V[t] + J[t]@((Vhat[t+1] - P[t])@J[t].T)
            
    paircov_prev = np.zeros((T,p,p))
    paircov_curr = np.zeros((T,p,p))
    for t in range(T):
        if t > 0:
            paircov_prev[t] = J[t-1]@Vhat[t] + np.outer(muhat[:,t], muhat[:,t-1])
        paircov_curr[t] = Vhat[t] + np.outer(muhat[:,t], muhat[:,t])
        
    return muhat, Vhat, paircov_prev, paircov_curr


def kalman_smoother(x, params):
    ''' Smoother simply does forward and backward recursions '''
    # forward filter first
    mu, V, P, K, c = kalman_filter(x, params)
    loglike = np.log(c).sum()
    # backward smooth
    muhat, Vhat, _, _ = kalman_backward(x, mu, V, P, params)
    return muhat, Vhat, loglike, K