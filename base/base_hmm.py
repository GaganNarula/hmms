import numpy as np
import abc
from sklearn.base import BaseEstimator


class BaseHMM(metaclass=abc.ABCMeta):
    ''' 
    Base class for all HMMs. 
    
    Forces subclasses to have 
    '''
    
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def get_params(self):
        ''' 
        Get parameters 
        
        '''
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_params(self):
        '''
        Set parameters
        '''
        raise NotImplementedError
        
    @abc.abstractmethod
    def posterior_distr(self, y, **args):
        """ 
        Compute posterior distribution over hidden states
        
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    def loglikelihood(self, y):
        """ 
        Compute marginal likelihood of the sequence y.
        
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    def fit(self, Y):
        """ 
        Use E.M to fit a sequence of sequences Y. 
        
        """
        raise NotImplementedError
        
        