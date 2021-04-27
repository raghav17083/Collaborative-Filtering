In a typical machine learning-based coreference resolution system (Soon et al. , 2001; Ng and Cardie, 2002b; Yang et al. , 2003; Luo et al. , 2004), a statistical model is learned from training data and is used to measure how likely an anaphor 1 is coreferential to a candidate antecedent. 
(2001) or Ng and Cardie (2002b). 
This is different from (Ng and Cardie, 2002a; Ng, 2004) where their anaphoricty models are trained independently of the coreference model, and it is either used as a pre-filter, or its output is used as features in the coreference model. 
Duplicated Ng and Cardie (2002a) baseline. 
For instance, Ng and Cardie (2002a) train an anaphoricity classifier to determine whether a mention is anaphoric, and let an independentlytrained coreference system resolve only those mentions that are classified as anaphoric. 
