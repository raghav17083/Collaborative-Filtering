Following (Matusov et al. , 2004), we compute these local costs by interpolating state occupation probabilities from the source-to-target and target-to-source training of the HMM and IBM-4 models as trained by the GIZA++ toolkit (Och et al. , 2003). 
Word Alignments in both source-to-target and target-to-source directions are obtained using the Maximum A-Posteriori (MAP) framework (Matusov et al., 2004). 
(Matusov et al. , 2004) presented a model capable of modeling 1-toN and M-to-1 alignments (but not arbitrary M-toN alignments) which was bootstrapped from Model 4. 
Nevertheless, recently such posterior probabilities have been used in SMT word alignment system as an alternative to Viterbi decoding, and helped to improve the performance of such systems (Matusov et al. , 2004; Liang et al. , 2006). 
The final alignments are determined using cost matrices defined by the state occupation probabilities of the trained HMM (Matusov et al. , 2004). 
