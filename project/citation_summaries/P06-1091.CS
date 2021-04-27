They can be used for discriminative training of reordering models (Tillmann and Zhang, 2006). 
The algorithm is slightly different from other online training algorithms (Tillmann and Zhang, 2006; Liang et al. , 2006) in that we keep and update oracle translations, which is a set of good translations reachable by a decoder according to a metric, i.e. BLEU (Papineni et al. , 2002). 
Tillmann and Zhang (2006), Liang et al. 
Tillmann and Zhang (2006) use a BLEU oracle decoder for discriminative training of a local reordering model. 
Tillmann and Zhang (2006) present a procedure to directly optimize the global scoring function used by a phrasebased decoder on the accuracy of the translations. 
