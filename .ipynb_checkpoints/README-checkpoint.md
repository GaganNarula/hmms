# hmms
Hidden markov models with discrete and continuous hidden states and various emission distributions

Hidden markov models are models of a sequence of random variables, where the hidden state models the dynamics and observations are generated at each time step.

hmms : 
    -> base : 
        -> base.py : base class
        -> filter.py : forward recursion
        -> smooth.py : backward recursion
    -> gaussian_hmm : discrete hidden states
    -> linear_gaussian_hmm : continous hidden state