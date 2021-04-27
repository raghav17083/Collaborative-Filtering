Vijay-Shanker and Joshi (1985) introduced the first TAG parser in a CYK-like algorithm. 
A conventional parsing algorithm (VijayShanker and Joshi, 1985) views the trees as independent, and so is likely to duplicate tile processing of this common structure. 
Tree Adjoining Grammars The Tree Adjoining Grammar formalism is a tree-generating formalism introduced in Joshi, Levy, and Takahashi (1975), linguistically motivated (see, for example, AbeillO et al. \[1990\] and Kroch and Joshi \[1985\]), and with formal properties studied in VijayShanker and Joshi (1985) and Vijay-Shanker and Weir (1994a, 1994b). 
4.1 TAG recognition We start with the discussion of a baseline recognition algorithm for TAG, along the lines of (VijayShanker and Joshi, 1985). 
The studied algorithm recognizes auxiliary trees without considering the substring dominated by the foot node, as is the case of the CYK-like algorithm in Vijay-Shanker and Joshi (1985). 
