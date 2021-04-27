(2000) and Martinez and Agirre (2000) conducted experiments using the DSO corpus, which contains sentences from two different corpora, namely Brown Corpus (BC) and Wall Street Journal (WSJ).


uating WSD systems on data drawn from different domains, several prior research efforts (Escudero et al., 2000; Martinez and Agirre, 2000) observed a similar drop in performance of about 10% when a WSD system that was trained on the BC part of the DSO corpus was evaluated on the WSJ part of the corpus, and vice versa.


All features encountered in the training data are ranked in the DL (best evidence first) according to the following loglikelihood ratio (Yarowsky, 1995): Log Pr(reading i jfeature k ) P j6=i Pr(reading j jfeature k ) We estimated probabilities via maximum likelihood, adopting a simple smoothing method (Martinez and Agirre, 2000): 0.1 is added to both the denominator and numerator.


