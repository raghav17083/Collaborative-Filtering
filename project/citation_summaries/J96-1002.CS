The disambiguation model of this parser is based on a maximum entropy model (Berger et al. , 1996). 
For the classifier, we used the OpenNLP MaxEnt implementation (maxent.sourceforge.net) of the maximum entropy classification algorithm (Berger et al. 1996). 
5.2 Maximum Entropy Model We use the Maximum Entropy (ME) Model (Berger et al. , 1996) for our classification task. 
(1996), Warnke et al. 
One is how to learn a statistical model to estimate the conditional probability    , and the other is how to generate confusion set C of a given query q 4.1 Maximum Entropy Model for Query Spelling Correction We take a feature-based approach to model the posterior probability     . Specifically we use the maximum entropy model (Berger et al. , 1996) for this task:     = exp     ,   =1 exp(     (,  ) =1 ) (2) where exp(     (, ) =1 ) is the normalization factor;   , is a feature function defined over query q and correction candidate c, while   is the corresponding feature weight. 
