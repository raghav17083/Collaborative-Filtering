This type of model is used by the majority of transition-based parsers (Attardi et al. , 2007; Duan et al. , 2007; Hall et al. , 2007a; Johansson and Nugues, 2007b; Mannem, 2007; Titov and Henderson, 2007; Wu et al. , 2007).


5.2.2 Inference The most common inference technique in transitionbased dependency parsing is greedy deterministic search, guided by a classifier for predicting the next transition given the current parser state and history, processing the tokens of the sentence in sequential left-to-right order7 (Hall et al. , 2007a; Mannem, 2007; Marinov, 2007; Wu et al. , 2007).


Two other entries used MIRA (Mannem, 2007) or online passive-aggressive learning (Johansson and Nugues, 2007b) to train a globally normalized model.


