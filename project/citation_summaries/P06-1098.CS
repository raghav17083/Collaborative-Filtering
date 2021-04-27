Based on hierarchical phrase-based modeling, we adopted the left-to-right target generation method (Watanabe et al. , 2006b). 
In the hierarchical phrase-based model, decoding is realized as an Earley-style top-down parser on the foreign language side with a beam search strategy synchronized with the cardinality of already translated foreign words (Watanabe et al. , 2006). 
Thus, we use an approximated BLEU score that basically computes BLEU for a sentence set, but accumulates the difference for a particular sentence (Watanabe et al. , 2006a). 
As a baseline SMT system, we use the hierarchical phrase-based translation with an efficient left-to-right generation (Watanabe et al. , 2006b) originally proposed by Chiang (2005). 
We applied an Earley-style top-down parsing approach (Wu and Wong, 1998; Watanabe et al. , 2006b; Zollmann and Venugopal, 2006). 
