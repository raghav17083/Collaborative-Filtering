Huang and Chiang (2005) define a forest as a tupleV,E, v,R, where V is a finite set of nodes, E is a finite set of hyperedges, vV is a distinguished node that denotes the goal item in parsing, and R is the set of weights. 
The U-best list is generated using Algorithm 3 of Huang and Chiang (2005). 
Using the k-best extraction algorithm (Huang and Chiang, 2005), we extract the k most likely derivations from the hypergraph. 
In a nutshell, cube pruning works on the LM forest, keeping at most k +LM items at each node, and uses the k-best parsing Algorithm 2 of Huang and Chiang (2005) to speed up the computation. 
We apply Algorithm 3 of Huang and Chiang (2005) for n-best list generation. 
