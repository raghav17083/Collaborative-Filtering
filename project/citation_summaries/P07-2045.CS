We use the MOSES decoder (Koehn et al., 2007) as a baseline. 
220 (Koehn et al., 2003); they can overlap.5 Additionally, since phrase features can be any function of words and alignments, we permit features that consider phrase pairs in which a target word outside the target phrase aligns to a source word inside the source phrase, as well as phrase pairs with gaps (Chiang, 2005; Ittycheriah and Roukos, 2007). 
We used Moses (Koehn et al., 2007), a phrase-based SMT toolkit, for training the translation model (and later for decoding). 
4 Evaluation Our baseline is a phrase-based MT system trained using the MOSES toolkit (Koehn et al. , 2007). 
The translation system is a factored phrasebased translation system that uses the Moses toolkit (Koehn et al., 2007) for decoding and training, GIZA++ for word alignment (Och and Ney, 2003), and SRILM (Stolcke, 2002) for language models. 
