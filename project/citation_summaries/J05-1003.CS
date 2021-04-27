At present, we use a feature set that is similar to the one used by Collins and Koo (2005). 
Parse reranking techniques, for instance, rely on knowledge about features other than those found in the core parsing model in order to determine the best parse (e.g., Collins and Koo, 2005; Charniak and Johnson, 2005). 
This leads to the discriminative re-ranking approach developed by Collins and Koo (2005), in which a statistical parser trained on a treebank is used to produce a set of candidate parses which are then re-ranked by an Abney-style probabilistic model. 
We use the following notation, adapted from Collins and Koo (2005):  Training data is a set of example input/output pairs. 
They rerank the n-best parses that are output by a generative parser using a lot of lexical and syntactic features (Collins and Koo, 2005; Charniak and Johnson, 2005). 
