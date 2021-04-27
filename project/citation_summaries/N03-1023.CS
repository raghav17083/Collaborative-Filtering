P09-1045:1	168:260	Bagging was used by Ng and Cardie (2003) to create committees of classifiers for labelling unseen data for retraining.
---------------------------------------------------
D08-1106:2	46:213	A major drawback of bootstrapping is the lack of principled method for selecting optimal parameter values (Ng and Cardie, 2003; Banko and Brill, 2001).
---------------------------------------------------
P09-2089:3	10:55	Co-training is one of thosemethods,andhasbeenextensivelystudiedin NLP(NigamandGhani,2000; PierceandCardie, 2001; Ng and Cardie, 2003; Mota and Grishman, 2008).
---------------------------------------------------
P05-1001:4	131:242	2 We also tested "self-training with bagging", which Ng and Cardie (2003) used for co-reference resolution.
---------------------------------------------------
P07-1125:5	40:186	(2000) for text categorization and by Ng and Cardie (2003) in combination with a meta-level feature selection procedure.
---------------------------------------------------
P07-1125:6	37:186	Co-training has also been used for named entity recognition (NER) (Collins and Singer, 1999), coreference resolution (Ng and Cardie, 2003), text categorization (Nigam and Ghani, 2000) and improving gene name data (Wellner, 2005).
---------------------------------------------------
W03-1015:7	98:197	See Ng and Cardie (2003) for details.
---------------------------------------------------
W03-1015:8	84:197	Unlike Ng and Cardie (2003) where we choose one of the dryrun texts (contributing approximately 35003700 instances) form the labeled data set, however, here we randomly select 1000 instances.
---------------------------------------------------
W03-1015:9	161:197	Our hypothesis is that selection methods that are based solely on the confidence assigned to an instance by a single classifier 5 Note that this is self-training without bagging, unlike the self-training algorithm discussed in Ng and Cardie (2003).
---------------------------------------------------
W03-1015:10	25:197	In related work (Ng and Cardie, 2003), we compare the performance of the Blum and Mitchell cotraining algorithm with that of two existing singleview bootstrapping algorithms  self-training with bagging (Banko and Brill, 2001) and EM (Nigam et al. , 2000)  on coreference resolution, and show that single-view weakly supervised learners are a viable alternative to co-training for the task.
---------------------------------------------------
W03-1015:11	158:197	Second, 1000 instances may simply not be sufficient for co-training to be effective for this task: in related work (Ng and Cardie, 2003), we find that starting with 35003700 labeled instances instead of 1000 allows co-training to improve the baseline by 4.6% and 9.5% in F-measure using naive Bayes classifiers for the MUC-6 and MUC-7 data sets, respectively.
---------------------------------------------------
W03-1015:12	28:197	(2003a), ranks unlabeled instances to be added to the labeled data in an attempt to alleviate a problem commonly observed in bootstrapping experiments  performance deterioration due to the degradation in the quality of the labeled data as bootstrapping progresses (Pierce and Cardie, 2001; Riloff and Jones, 1999).
---------------------------------------------------
D09-1033:13	134:223	A similar strategy was employed by Ng and Cardie (2003) in a self-training set-up.
---------------------------------------------------
P09-1027:14	88:195	Till now, co-training has been successfully applied to statistical parsing (Sarkar, 2001), reference resolution (Ng and Cardie, 2003), part of speech tagging (Clark et al., 2003), word sense disambiguation (Mihalcea, 2004) and email classification (Kiritchenko and Matwin, 2001).
---------------------------------------------------
W04-2405:15	46:194	a8 Growth size (G) Number of most confidently labeled examples that are added at each iteration to the set of labeled data L. As previously noticed (Ng and Cardie, 2003), there is no principled method for selecting optimal values for these parameters, which is an important disadvantage of these algorithms.
---------------------------------------------------
W04-2405:16	28:194	(Ng and Cardie, 2003) define self-training as a single-view weakly supervised algorithm, build by training a committee of classifiers using bagging, combined with majority voting for final label selection.
---------------------------------------------------
W04-2405:17	167:194	This is fundamentally different from the approach proposed in (Ng and Cardie, 2003), where they also apply majority voting in a bootstrapping framework, but in a different setting.
---------------------------------------------------
W04-2405:18	20:194	In natural language learning, co-training was applied to statistical parsing (Sarkar, 2001), reference resolution (Ng and Cardie, 2003), part of speech tagging (Clark et al. , 2003), and others, and was generally found to bring improvement over the case when no additional unlabeled data are used.
---------------------------------------------------
W05-0612:19	35:219	As an alternative to co-training, Ng and Cardie (2003) use EM to augment a supervised coreference system with unlabeled data.
---------------------------------------------------
W09-2211:20	20:28	While there are quite a few success stories reporting considerable performance gains over an inductive baseline (e.g., parsing (McClosky et al., 2008), coreference resolution (Ng and Cardie, 2003), and machine translation (Ueffing et al., 2007)), there are negative results too (see Pierce and Cardie (2001), He and Gildea (2006), Duh and Kirchhoff (2006)).
---------------------------------------------------
D08-1067:21	15:242	However, classifying mention pairs using such iterative approaches is undesirable for coreference resolution: since the non-coreferent mention pairs significantly outnumber their coreferent counterparts, the resulting classifiers generally have an increasing tendency to (mis)label a pair as non-coreferent as bootstrapping progresses (see Ng and Cardie (2003)).
---------------------------------------------------
D08-1067:22	13:242	In fact, several popular weakly supervised learning algorithms such as self-training, co-training (Blum and Mitchell, 1998), and EM (Dempster et al., 1977) have been applied to coreference resolution (Ng and Cardie, 2003) and the related task of pronoun resolution (Muller et al., 2002; Kehler et al., 2004; Cherry and Bergsma, 2005).
---------------------------------------------------
