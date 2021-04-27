C08-1051:1	7:191	Furthermore, recent studies revealed that word clustering is useful for semi-supervised learning in NLP (Miller et al., 2004; Li and McCallum, 2005; Kazama and Torisawa, 2008; Koo et al., 2008).
---------------------------------------------------
D08-1042:2	78:214	There has been a lot of progress in learning dependency tree parsers (McDonald et al., 2005; Koo et al., 2008; Wang et al., 2008).
---------------------------------------------------
D09-1004:3	15:221	The results from CoNLL shared tasks in 2005 and 2008 (Carreras and Marquez, 2005; Koomen et al., 2005; Surdeanu et al., 2008; Johansson and Nugues, 2008), further show that SRL pipeline may be one of the standard to achieve a state-of-the-art performance in practice.
---------------------------------------------------
D09-1004:4	68:221	These works include (Gildea and Jurafsky, 2002; Carreras and Marquez, 2005; Koomen et al., 2005; Marquez et al., 2005; Dang and Palmer, 2005; Pradhan et al., 2005; Toutanova et al., 2005; Jiang and Ng, 2006; Liu and Ng, 2007; Surdeanu et al., 2007; Johansson and Nugues, 2008; Che et al., 2008).
---------------------------------------------------
W08-2102:5	194:209	The method shows improvements over the method described in (Koo et al., 2008), which is a state-of-the-art second-order dependency parser similar to that of (McDonald and Pereira, 2006), suggesting that the incorporation of constituent structure can improve dependency accuracy.
---------------------------------------------------
W08-2102:6	32:209	The model also recovers dependencies with significantly higher accuracy than state-of-the-art dependency parsers such as (Koo et al., 2008; McDonald and Pereira, 2006).
---------------------------------------------------
W08-2102:7	163:209	KCC08 unlabeled is from (Koo et al., 2008), a model that has previously been shown to have higher accuracy than (McDonald and Pereira, 2006).
---------------------------------------------------
W08-2102:8	164:209	KCC08 labeled is the labeled dependency parser from (Koo et al., 2008); here we only evaluate the unlabeled accuracy.
---------------------------------------------------
D09-1058:9	171:216	It is clear that the gains from our method are larger for smaller labeled data sizes, a tendency that was also observed in (Koo et al., 2008).
---------------------------------------------------
D09-1058:10	157:216	5 Results and Discussion Table 3 gives results for the SS-SCM method under various configurations: for first and secondorder parsing models, with and without the cluster features of (Koo et al., 2008), and for varying amounts of labeled data.
---------------------------------------------------
D09-1058:11	10:216	It is often straightforward to obtain large amounts of unlabeled data, making semi-supervised approaches appealing; previous work on semisupervised methods for dependency parsing includes (Smith and Eisner, 2007; Koo et al., 2008; Wang et al., 2008).
---------------------------------------------------
D09-1058:12	118:216	as those described in (McDonald et al., 2005a; McDonald et al., 2005b; McDonald and Pereira, 2006; Koo et al., 2008).
---------------------------------------------------
D09-1058:13	135:216	4.2.2 Cluster-based Features In a second set of experiments, we make use of the feature set used in the semi-supervised approach of (Koo et al., 2008).
---------------------------------------------------
D09-1058:14	145:216	These settings match the evaluation setting in previous work such as (McDonald et al., 2005a; Koo et al., 2008).
---------------------------------------------------
D09-1058:15	19:216	The first extension is to combine our method with the cluster-based semi-supervised method of (Koo et al., 2008).
---------------------------------------------------
D09-1058:16	4:216	Moreover, we introduce two extensions related to dependency parsing: The first extension is to combine SS-SCMs with another semi-supervised approach, described in (Koo et al., 2008).
---------------------------------------------------
D09-1058:17	132:216	Our baseline features (baseline) are very similar to those described in (McDonald et al., 2005a; Koo et al., 2008): these features track word and POS bigrams, contextual features surrounding dependencies, distance features, and so on.
---------------------------------------------------
D09-1058:18	95:216	We simply use the clusterbased feature-vector representation f(x,y) introduced by (Koo et al., 2008) as the basis of our approach.
---------------------------------------------------
D09-1058:19	147:216	Since this method only considers projective dependency structures, we projectivized the PDT training data in the same way as (Koo et al., 2008).
---------------------------------------------------
D09-1058:20	124:216	These data sets are identical to the unlabeled data used in (Koo et al., 2008), and are disjoint from the training, development and test sets.
---------------------------------------------------
D09-1058:21	200:216	(a) English dependency parsers on PTB dependency parser test description (McDonald et al., 2005a) 90.9 1od (McDonald and Pereira, 2006) 91.5 2od (Koo et al., 2008) 92.23 1od, 43M ULD SS-SCM (w/ CL) 92.70 1od, 3.72G ULD (Koo et al., 2008) 93.16 2od, 43M ULD 2-stage SS-SCM(+MIRA, w/ CL) 93.79 2od, 3.72G ULD (b) Czech dependency parsers on PDT dependency parser test description (McDonald et al., 2005b) 84.4 1od (McDonald and Pereira, 2006) 85.2 2od (Koo et al., 2008) 86.07 1od, 39M ULD (Koo et al., 2008) 87.13 2od, 39M ULD SS-SCM (w/ CL) 87.14 1od, 39M ULD 2-stage SS-SCM(+MIRA, w/ CL) 88.05 2od, 39M ULD Table 6: Comparisons with the previous top systems: (1od, 2od: 1stand 2nd-order parsing model, ULD: unlabeled data).
---------------------------------------------------
W09-0905:22	7:202	These methods are successful and useful (e.g. Koo et al., 2008), but in both cases it is not always clear whether errors in lexical classification are due to a problem in the induction algorithm or in what contexts count as identifying the same category (cf.
---------------------------------------------------
W09-0905:23	17:202	Evaluationofinductionmethodsisdifficult, dueto the variety of corpora and tagsets in existence (see discussion in Clark, 2003) and the variety of potential purposes for induced categories (e.g., Koo et al., 2008; Miller et al., 2004).
---------------------------------------------------
P09-1007:24	41:206	The first is usually focus on exploiting automatic generated labeled data from the unlabeled data (Steedman et al., 2003; McClosky et al., 2006; Reichart and Rappoport, 2007; Sagae and Tsujii, 2007; Chen et al., 2008), the second is on combining supervised and unsupervised methods, and only unlabeled data are considered (Smith and Eisner, 2006; Wang and Schuurmans, 2008; Koo et al., 2008).
---------------------------------------------------
W09-1104:25	47:228	Research in the field of unsupervised and weakly supervised parsing ranges from various forms of EM training (Pereira and Schabes, 1992; Klein and Manning, 2004; Smith and Eisner, 2004; Smith and Eisner, 2005) over bootstrapping approaches like selftraining (McClosky et al., 2006) to feature-based enhancements of discriminative reranking models (Koo et al., 2008) and the application of semisupervised SVMs (Wang et al., 2008).
---------------------------------------------------
W09-1119:26	140:223	In this work, we analyze a simple technique of using word clusters generated from unlabeled text, which has been shown to improve performance of dependency parsing (Koo et al., 2008), Chinese word segmentation (Liang, 2005) and NER (Miller et al., 2004).
---------------------------------------------------
D09-1087:27	26:227	There are other successful semi-supervised training approaches for dependency parsing, such as (Koo et al., 2008; Wang et al., 2008), and it would be interesting to investigate how they could be applied to constituency parsing.
---------------------------------------------------
D09-1060:28	116:253	To match previous work (McDonald et al., 2005; McDonald and Pereira, 2006; Koo et al., 2008), we split the data into a training set (sections 2-21), a development set (Section 22), and a test set (section 23).
---------------------------------------------------
D09-1060:29	148:253	(2007), Z&C 2008 refers to the combination graph-based and transition-based system of Zhang and Clark (2008), KOO08-dep1c/KOO08dep2c refers to a graph-based system with first/second-order cluster-based features by Koo et al.
---------------------------------------------------
D09-1060:30	30:253	We also demonstrate that our approach and other improvement techniques (Koo et al., 2008; Nivre and McDonald, 2008) are complementary and that we can achieve very high accuracies when we combine our method with other improvement techniques.
---------------------------------------------------
P09-1116:31	19:254	This method has been shown to be quite successful in named entity recognition (Miller et al. 2004) and dependency parsing (Koo et al., 2008).
---------------------------------------------------
P09-1116:32	29:254	Previous approaches, e.g., (Miller et al. 2004) and (Koo et al. 2008), have all used the Brown algorithm for clustering (Brown et al. 1992).
---------------------------------------------------
D09-1160:33	9:244	The feature combinations play an essential role in obtaining a classifier with state-of-the-art accuracy for several NLP tasks; recent examples include dependency parsing (Koo et al., 2008), parse re-ranking (McClosky et al., 2006), pronoun resolution (Nguyen and Kim, 2008), and semantic role labeling (Liu and Sarkar, 2007).
---------------------------------------------------
