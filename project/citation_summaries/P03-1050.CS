As in (Rogati et al. , 2003), we can find the most probable stemming given the model, apply this stemming, and retrain our word alignment system.


We tried the best performing scoring heuristic for Arabic from (Rogati et al. , 2003) where p(sj,z, aj|e) is modeled using the heuristic p(sj,z|lj) where sj,z is the Romanian suffix, and lj is the last letter of the Romanian word fj; these adjustments are updated during EM training.


(Rogati et al. , 2003) extended Model 1 with an additional hidden variable to represent the split points in Arabic between the prefix, the stem and the suffix to generate a stemming for use in Cross-Lingual Information Retrieval.


In fact, Rogati et al., (2003) employ this method to learn arabic morphology assuming annotations provided by an English stemmer.


