4.3 Baselines 4.3.1 Word Alignment We used the GIZA++ implementation of IBM word alignment model 4 (Brown et al., 1993; Och and Ney, 2003) for word alignment, and the heuristics described in (Och and Ney, 2003) to derive the intersection and refined alignment. 
We use GIZA++ (Och and Ney, 2003) to train generative directed alignment models: HMM and IBM Model4 (Brown et al., 1993) from training record-text pairs. 
We extract a phrase table using the Moses pipeline, based on Model 4 word alignments generated from GIZA++ (Och and Ney, 2003). 
The first is to align the words using a standard word alignement technique, such as the Refined Method described in (Och and Ney, 2003) (the intersection of two IBM Viterbi alignments, forward and reverse, enriched with alignments from the union) and then generate bi-phrases by combining together individual alignments that co-occur in the same pair of sentences. 
We used GIZA++ package (Och and Ney, 2003) to train IBM translation models. 
