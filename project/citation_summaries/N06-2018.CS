A few methods have been proposed, based mostly on the conventions of uncertainty sampling, where the learner queries the instance about which it has the least certainty (Scheffer et al., 2001; Culotta and McCallum, 2005; Kim et al., 2006), or query-by-committee, where a committee of models selects the instance about which its members most disagree (Dagan and Engelson, 1995).


Thus an alternate query strategy is sequence entropy (SE): SE(x) =  summationdisplay y P(y|x;)logP(y|x;), (3) where y ranges over all possible label sequences for input sequence x. Note, however, that the number 1072 of possible labelings grows exponentially with the length of x. To make this feasible, previous work (Kim et al., 2006) has employed an approximation we call N-best sequence entropy (NSE): NSE(x) =  summationdisplay yN P(y|x;)logP(y|x;), where N = {y1,,yN}, the set of the N most likely parses, and the posteriors are re-normalized (i.e., Z(x) in Equation (1) only ranges over N).


The measure can be also used as a confidence estimator in active learning in CRFs (Kim et al. , 2006), where examples with the most uncertainty are selected for presentation to human annotators.


