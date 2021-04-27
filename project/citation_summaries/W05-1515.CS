We then determine the accuracy of the parser on a held-out development set using the previous  value (before it was decreased), and stop training when this 2Turian and Melamed (2005) reported that decision trees applied to parsing have higher accuracy and training speed than decision stumps, so we build full decision trees rather than stumps. 
(2004) and Turian and Melamed (2005) for their discriminative parsers, which were also trained and tested on  15 word sentences. 
20 Turian and Melamed (2005) observed that uniform example biases b(i) produced lower accuracy as training progressed, because the induced classifiers minimized the example-wise error. 
% Turian and Melamed (2005) 87.13 86.47 87.80 Bikel (2004) 88.30 87.85 88.75 Taskar et al. 
In this way, instead of choosing the best  heuristically, we can optimize it during a single training run (Turian & Melamed, 2005). 
