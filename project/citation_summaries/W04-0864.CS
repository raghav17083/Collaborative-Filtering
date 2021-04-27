Another top ranked system is the one developed by (Yuret, 2004), which combines two Naive Bayes statistical models, one based on surrounding collocations and another one based on a bag of words around the target word.


(Yuret, 2004) observed that approximately half of the test instances do not match any of the contextual features learned from the training data for an all words disambiguation task.


We consider the outputs of the top 3 allwords WSD systems that participated in Senseval-3: Gambl (Decadt et al. , 2004), SenseLearner (Mihalcea and Faruque, 2004), and KOC University (Yuret, Nouns Verbs Adjectives F-SCORE 0.4228 0.4319 0.4727 Feature F-Score Ablation Difference TOPSIG 0.0403   OED 0.0355 0.0126 -0.0124 DERIV 0.0351 0.0977 0.0352 RES 0.0287 0.0147  TWIN 0.0285 0.0109 -0.0130 MN 0.0188 0.0358  LESK 0.0183 0.0541 -0.0250 SENSENUM 0.0155 0.0146 -0.0147 SENSECNT 0.0121 0.0160 0.0168 DOMAIN 0.0119 0.0082 -0.0265 LCH 0.0099 0.0068  WUP 0.0036 0.0168  JCN 0.0025 0.0190  ANTONYM 0.0000 0.0295 0.0000 MAXMN -0.0013 0.0179  VEC -0.0024 0.0371 -0.0062 HSO -0.0073 0.0112 -0.0246 LIN -0.0086 0.0742  COUSIN -0.0094   VERBGRP  0.0327  VERBFRM  0.0102  PERTAINYM   -0.0029 Table 4: Feature ablation study; F-score difference obtained by removal of the single feature 2004).


F1 F1flne Gambl 0.779 0.779 0.779 0.652 SenseLearner 0.769 0.769 0.769 0.646 KOC Univ. 0.768 0.768 0.768 0.641 SSI 0.758 0.758 0.758 0.612 IRST-DDD 0.721 0.719 0.720 0.583 FS baseline 0.769 0.769 0.769 0.624 Random BL 0.497 0.497 0.497 0.340 University (Yuret, 2004)  and the best unsupervised system, namely IRST-DDD (Strapparava et al. , 2004).


