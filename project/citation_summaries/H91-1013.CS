3.1.1 Recognition Paradigm We used the N-best rescoring formalism for recognition experiments with the test data (Ostendorf, et al. 1991).


6 Weights Optimization The optimization of the system and feature weights may be carried out using a63 -best lists as in (Ostendorf et al. , 1991).


3 Experiments We designed our experiments to show the impact of each feature separately as well as their cumulative impact:  BASE: baseline string-to-dependency system  SLM: baseline + source dependency LM  CLM: baseline + context LM  LEN: baseline + length distribution  LBL: baseline + syntactic labels  LBL+LEN: baseline + syntactic labels + length distribution  LBL+LEN+CLM: baseline + syntactic labels + length distribution + context LM All the models were optimized on lower-cased IBM BLEU with Powells method (Powell, 1964; Brent, 1973) on n-best translations (Ostendorf et al., 1991), but evaluated on both IBM BLEU and TER.


