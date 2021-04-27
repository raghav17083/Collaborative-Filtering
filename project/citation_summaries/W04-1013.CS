We evaluate the system generated summaries using the automatic evaluation toolkit ROUGE (Lin, 2004). 
In this paper, we compare the performance of this system, HybridTrim, with the Topiary system and a number of other baseline gisting systems on a collection of news documents from the DUC 2004 corpus (DUC, 2003). 
Empirical evaluations using two standard summarization metricsthe Pyramid method (Nenkova and Passonneau, 2004b) and ROUGE (Lin, 2004)show that the best performing system is a CRF incorporating both order-2 Markov dependencies and skip-chain dependencies, which achieves 91.3% of human performance in Pyramid score, and outperforms our best-performing non-sequential model by 3.9%. 
ROUGE (Lin, 2004) is an evaluation metric designed to evaluate automatically generated summaries. 
Therefore, Lin and Och (2004) introduced skip-bigram statistics for the evaluation of machine translation. 
