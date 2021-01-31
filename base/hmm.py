import numpy as np
import abc


class BaseHMM(metaclass = abc.ABCMeta):
    ''' Base class for all HMMs. '''
    
    def forward_pass(self):
        return
    
    def backward_pass(self):
        return
    
    def loglikelihood(self):
        return