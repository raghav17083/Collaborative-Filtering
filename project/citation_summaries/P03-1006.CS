3.3 Representation of n-gram language models An n-gram model can be efficiently represented in a deterministic WFA, through the use of failure transitions (Allauzen et al. , 2003). 
The conditional probabilities and backoff structure defined in equation (5) can be encoded as a weighted finite state acceptor (Allauzen et al., 2003). 
This concept lattice is rescored with a Language Model on the concepts (also encoded as FSMs with the AT&T GRM toolkit (Allauzen et al. , 2003)). 
We use the offline approximation in which failure transitions are replaced with epsilons (Allauzen et al., 2003). 
Theses grammars are represented by Finite State Machines (FSMs) (thanks to the AT&T GRM/FSM toolkit (Allauzen et al. , 2003)). 
