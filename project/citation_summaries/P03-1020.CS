It is also called truecasing (Lita et al. , 2003). 
The capitalization problem can be seen as a sequence tagging problem (Chelba and Acero, 2004; Lita et al., 2003; Kim and Woodland, 2004), where each lower-case word is associated to a tag that describes its capitalization form. 
The trigram-based baseline is similar to the one in (Lita et al. , 2003) except that we used Kneser-Ney smoothing instead of a mixture. 
(Lita et al., 2003) builds a trigram language model (LM) with pairs (word, tag), estimated from a corpus with case information, and then uses dynamic programming to disambiguate over all possible tag assignments on a sentence. 
A 1-gram capitalizer is usually used as a baseline for capitalization experiments (Lita et al. , 2003; Kim and Woodland, 2004; Chelba and Acero, 2004). 
