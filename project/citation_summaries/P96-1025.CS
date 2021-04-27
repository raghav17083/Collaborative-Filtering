This is a modified version of the backed-off smoothing used by Collins (1996) to alleviate sparse data problems. 
Collins (Collins, 1996) used dependencies as the backbone for his probabilistic parser and there has been work on learning both probabilistic (Carroll, 1992; Lee, 1999; Paskin, 2001) and transformation based dependency grammars (Hajic, 1997). 
3 The Probability Model The DAG-like nature of the dependency structures makes it difficult to apply generative modelling techniques (Abney, 1997; Johnson et al. , 1999), so we have defined a conditional model, similar to the model of Collins (1996) (see also the conditional model in Eisner (1996b)). 
Collins (personal communication) independently observed the usefulness of this modification, and Caraballo and Charniak (1996) used a related technique in a best-first parser. 
Collins (1996) proposed a statistical parser which is based on probabilities of dependencies between head-words in the parse tree. 
