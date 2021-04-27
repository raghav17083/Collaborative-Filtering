3 Incremental Parsing Method Based on Adjoining Operation In order to avoid the problem of infinite local ambiguity, the previous works have adopted the following approaches: (1) a beam search strategy (Collins and Roark, 2004; Roark, 2001; Roark, 2004), (2) limiting the allowable chains to those actually observed in the treebank (Collins and Roark, 2004), and (3) transforming the parse trees with a selective left-corner transformation (Johnson and Roark, 2000) before inducing the allowable chains and allowable triples (Collins and Roark, 2004).


As a practical matter, the set of leftchild chains for a terminal x is taken to be the union of the sets of left-child chains for all pre-terminal part-ofspeech (POS) tags T for x. Before inducing the left-child chains and allowable triples from the treebank, the trees are transformed with a selective left-corner transformation (Johnson and Roark, 2000) that has been flattened as presented in Roark (2001b).


To avoid some of the early prediction of structure, the version of the Roark parser that we used 326 performs an additional grammar transformation beyond the simple factorization already described  a selective left-corner transform of left-recursive productions (Johnson and Roark, 2000).


The trees are transformed with a selective left-corner transformation (Johnson and Roark, 2000) that has been flattened as presented in Roark (2001b).


