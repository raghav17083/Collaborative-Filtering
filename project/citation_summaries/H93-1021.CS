The supervised methods are based on Maximum Entropy (ME) (Lau et al. , 1993; Berger et al. , 1996; Ratnaparkhi, 1998), neural network using the Learning Vector Quantization algorithm (Kohonen, 1995) and Specialized Hidden Markov Models (Pla, 2000).


47 models, thatlogZ = (Z)/Z = r/Z, the vector of feature expectations (Lau et al., 1993).


Previous uses of this model include language modeling(Lau et al. , 1993), machine translation(Berger et al. , 1996), prepositional phrase attachment(Ratnaparkhi et al. , 1994), and word morphology(Della Pietra et al. , 1995).


In practice, 7-/ is very large and the model's expectation Efj cannot be computed directly, so the following approximation(Lau et al. , 1993) is used: n E fj,~ E15(hi)p(tilhi)fj(hi,ti) i=1 where fi(hi) is the observed probability of the history hi in the training set.


Our systems use both corpus-based and knowledge-based approaches: Maximum Entropy(ME) (Lau et al. , 1993; Berger et al. , 1996; Ratnaparkhi, 1998) is a corpus-based and supervised method based on linguistic features; ME is the core of a bootstrapping algorithm that we call re-training inspired  This paper has been partially supported by the Spanish Government (CICyT) under project number TIC-2003-7180 and the Valencia Government (OCyT) under project number CTIDIB-2002-151 by co-training (Blum and Mitchell, 1998); Relevant Domains (RD) (Montoyo et al. , 2003) is a resource built from WordNet Domains (Magnini and Cavaglia, 2000) that is used in an unsupervised method that assigns domain and sense labels; Specification Marks(SP) (Montoyo and Palomar, 2000) exploits the relations between synsets stored in WordNet (Miller et al. , 1993) and does not need any training corpora; Commutative Test (CT) (Nica et al. , 2003), based on the Sense Discriminators device derived from EWN (Vossen, 1998), disambiguates nouns inside their syntactic patterns, with the help of information extracted from raw corpus.


