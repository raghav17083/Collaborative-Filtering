The plausibility is a measure of the statistical evidence of a detected syntactic link (Basili et al, 1994b), that depends upon local (i.e. at the sentence level) syntactic ambiguity and global corpus evidence.


The shallow parser (Basili et al. 1994), extracts from the learning corpus elementary syntactic relations such as subject-object, noun-preposition-noun, etc. A syntactic link (hereafter esl) is represented as: esli(wj, mod(typei, wk)) where w i is the head word, Wk is the modifier, hnd typei is the type of syntactic relation (e.g. PP(of), PP(for), SUB J-Verb, Verb-DirectObject, etc.).


However, the parser assigns to each detected esl a statistical measure of confidence, called plausibility (Basili et al. 1994b).


To conduct the experiment, we used a shallow syntactic analyzer (SSA) (Basili et al, 1994) to extract word associations from two very different corpora in Italian (a scientific corpus of environmental abstracts, called ENEA, and a legal corpus on taxation norms, called LD) 2 Given a corpus, SSA produces an extensive database of elementary syntactic links (esl).


Our approach can be opposed to the a prior one which consists in building simplified representations while parsing (Basili et al. , 1994; Metzler and Haas, 1989; Smeaton and Sheridan, 19911).


