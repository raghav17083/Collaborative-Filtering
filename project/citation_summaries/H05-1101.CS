Grammars with longer rules can represent a larger set of reorderings between languages (Aho and Ullman, 1972), but also require greater computational complexity for word alignment algorithms based on synchronous parsing (Satta and Peserico, 2005). 
We adopt the SCFG notation of Satta and Peserico (2005). 
2 Synchronous CFGs and permutation trees We begin by describing the synchronous CFG formalism, which is more rigorously de ned by Aho and Ullman (1972) and Satta and Peserico (2005). 
Because general SCFG parsing is an NPhard problem (Satta and Peserico, 2005), practical SMT decoders based on SCFG parsing requires an equivalent binary SCFG that is directly learned from training data to achieve polynomial time complexity using the CKY algorithm (Kasami, 1965; Younger, 1967) borrowed from CFG parsing techniques. 
We adopt the SCFG notation of Satta and Peserico (2005). 
