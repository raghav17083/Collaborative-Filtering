More recent studies try to solve the problem of POS induction by combining distributional and morphological information (Clark, 2003; Freitag, 2004), or by clustering words and projecting them to POS vectors (Rapp, 2005). 
3 Evaluation methodology We adopt the methodology of (Freitag 2004) and measure cluster-conditional tag perplexity PP as the average amount of uncertainty to predict the tags of a POS-tagged corpus, given the tagging with classes from the unsupervised method. 
More recent work includes Clark (2003) who combines distributional and morphological information, and Freitag (2004) who uses a hidden Marcov model in combination with co-clustering. 
Freitag (2004) uses the similar cluster-conditional tag perplexity which is merely exp(H(TG|TI))2. 
Freitag (2004) does not sum up the contexts of each word in a context vector, but the most frequent instances of four-word windows are used in a co-clustering algorithm. 
