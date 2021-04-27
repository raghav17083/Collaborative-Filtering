Figure3showstheresultsofperformingthiscombined precision/recall optimization on three separate n-best lists: the 50-best list of base-phrase trees extracted from the full-parse output of the Charniak and Johnson (2005) reranker; the 50-best list output by the Hollingshead et al.


We derive thesebase-phraseconstraintsfromthreesources: the reranking stage of the parsing pipeline; a finite-state shallow parser (Hollingshead et al. , 2005); and a combination of the output from these two sources.


We annotate tag sequences onto the word sequence via a competitive discriminatively trained tagger (Hollingshead et al. , 2005), trained for each of two kinds of tag sequences: part-of-speech (POS) tags and shallow parse tags.


The tagger (Hollingshead et al., 2005) uses log linear models trained with the perceptron algorithm, and derives, via the forward-backwardalgorithm, theposteriorprobability of each of the two tags at each word, so that Pr(B) + Pr(B) = 1.


