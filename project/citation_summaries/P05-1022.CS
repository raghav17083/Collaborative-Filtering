Parse reranking techniques, for instance, rely on knowledge about features other than those found in the core parsing model in order to determine the best parse (e.g., Collins and Koo, 2005; Charniak and Johnson, 2005). 
33 using the reranking parser of Charniak and Johnson (2005). 
(2005) 86.1 86.0 1.39 58.3 Collins (1999) 88.3 88.1 1.06 64.0 Charniak and Johnson (2005) 89.5 89.6 0.88 67.6 This Paper 89.8 89.6 0.92 66.3 Table 4: Comparison of our results with those of others. 
1 Introduction Probabilistic context-free grammars (PCFGs) underlie most high-performance parsers in one way or another (Collins, 1999; Charniak, 2000; Charniak and Johnson, 2005). 
Charniak and Johnson (2005) use a PCFG to do a pass of inside-outside parsing to reduce the state space of a subsequent lexicalized n-best parsing algorithm to produce parses that are further re-ranked by a MaxEnt model. 
