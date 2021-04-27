3.1 Chart Generation Algorithm The generation algorithm is based on chart generation as first introduced by Kay (1996) with Viterbi-pruning. 
We followed two partial solutions to this problem by Kay (1996). 
(Kay, 1996)) is to index edges with semantic indices (for instance, the edge with category N/x:dog(x) will be indexed with x) and to restrict edge combination to these edges which have compatible indices. 
4 Chart Generation and Smoothing Algorithms 4.1 Chart Generation Algorithm The PCFG-based generation algorithms are implemented in terms of a chart generator (Kay, 1996). 
Following Stone and Doran (1997) and Kay (1996), we enhance this TAG grammar with a syntax-semantics interface in which nonterminal nodes of the elementary trees are equipped with index variables, which can be bound to individuals in the semantic input. 
