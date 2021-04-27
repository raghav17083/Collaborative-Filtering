Further experiments using MBR for Bleu were performed in (Venugopal et al. , 2005; Ehling et al. , 2007).


4 Related Work Various methods are used to optimize log-linear models in re-ranking (Shen et al., 2004; Venugopal et al., 2005; Smith and Eisner, 2006).


All feature weights are automatically determined and optimized with respect to BLEU via MERT (Venugopal et al., 2005).


In order to find an optimal set of weights, we use MER training as described in (Venugopal et al., 2005), which uses rescoring of the top n hypotheses to maximize an evaluation metric like BLEU or TER.


Weights for each component score were optimized to maximize BLEU-score on the development set using MER optimization as described in (Venugopal et al. , 2005).


