There are two approaches to solve this problem: to increase the coverage of the dictionary (Fung and Wu, 1994; Chang et al. , 1995; Mori and Nagao, 1996) and to design a better model for unknown words (Nagata, 1996; Sproat et al. , 1996).


(Chang et al. , 1995) proposed an automatic dictionary construction method for Chinese from a large unsegmented corpus (311591 sentences) with the help of a small segmented seed corpus (1000 sentences).


It is impossible to compare our results with (Chang et al. , 1995), because the experiment conditions are completely different in terms of language (Chinese vs. Japanese), the size of seed segmented corpus, the size of target unsegmented corpus and its out-of-vocabulary rate, the size of initial word list, and the type of reference data 57 (on-line dictionary vs. segmented corpus).


\[Chang et al. , 1995\] combined a small seed segmented corpus and a large unsegmented corpus to build a word unigram model using the Viterbi re-estimation.


\[Chang et al. , 1995\] used a statistical method called "Two-Class Classifier", which decided whether the string is actually a word based on the features derived from character N-gram.


