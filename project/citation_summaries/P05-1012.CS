N07-3002:1	11:74	Currently, the work on conditional parsing models appears to have culminated in large margin training approaches (Taskar et al. , 2004; McDonald et al. , 2005), which demonstrates the state of the art performance in English dependency parsing.
---------------------------------------------------
N07-3002:2	20:74	In which case, the parsing problem reduces to a18a37a36 a1a39a38a41a40a43a42a45a44a46a38a48a47 a49a22a50a41a51a53a52a55a54a57a56 a58 a52a60a59a62a61a5a63a64a59a66a65a43a56a67a50a26a49 sa2a5a4a29a19a22a21a23a4a25a24a28a16 (1) where the score sa2a5a4 a19 a21 a4 a24 a16 can depend on any measurable property of a4a20a19 and a4a25a24 within the tree a18 . This formulation is suf ciently general to capture most dependency parsing models, including probabilistic dependency models (Wang et al. , 2005; Eisner, 1996) as well as non-probabilistic models (McDonald et al. , 2005; Wang et al. , 2006).
---------------------------------------------------
N07-3002:3	7:74	1 Introduction Over the past decade, there has been tremendous progress on learning parsing models from treebank data (Magerman, 1995; Collins, 1999; Charniak, 1997; Ratnaparkhi, 1999; Charniak, 2000; Wang et al. , 2005; McDonald et al. , 2005).
---------------------------------------------------
N07-3002:4	14:74	Another unexploited connection is that probabilistic approaches pay closer attention to the individual errors made by each component of a parse, whereas the training error minimized in the large margin approach the structured margin loss (McDonald et al. , 2005) is a coarse measure that only assesses the total error of an entire parse rather than focusing on the error of any particular component.
---------------------------------------------------
N07-3002:5	12:74	Despite the realization that maximum margin training is closely related to maximum conditional likelihood for conditional models (McDonald et al. , 2005), a suf ciently uni ed view has not yet been achieved that permits the easy exchange of improvements between the probabilistic and nonprobabilistic approaches.
---------------------------------------------------
N07-3002:6	30:74	The structured large margin approach, on the other hand, uses a global scoring function by minimizing a training loss the structured margin loss (McDonald et al. , 2005) which is directly coordinated with the global tree.
---------------------------------------------------
C08-1046:7	119:234	4 Experiment 4.1 Settings We implemented the tournament model, the CC algorithm (Kudo and Matsumoto, 2002), SR algorithm (Sassano, 2004) and CLE algorithm (McDonald et al., 2005) with SVM classifiers.
---------------------------------------------------
C08-1046:8	164:234	365 Method Features Jan. 9th Jan.10th Jan. 15th Tournament Standard feature only 89.89/49.63 89.63/48.34 89.40/49.70 All features 90.09/49.71 90.11/49.02 90.35/52.59 SR algorithm Standard feature only 88.18/45.92 88.80/44.76 88.03/47.24 (Sassano, 2004) All features 89.22/47.90 89.79/47.87 89.55/49.79 CC algorithm Standard feature only 88.17/45.92 88.80/44.76 88.00/47.24 (Kudo and Matsumoto, 2002) All features 89.22/47.90 89.80/47.94 89.53/49.79 CLE algorithm Standard feature only 88.64/45.34 88.16/43.14 88.07/45.21 (McDonald et al., 2005) Standard and Additional 89.21/46.83 89.05/45.03 88.90/48.43 Table 2: Dependency and sentence accuracy [%] using 7,587 sentences as training data.
---------------------------------------------------
P06-2009:9	36:229	(McDonald et al. , 2005) build on this work, and use a global discriminative training approach to improve the edges scores, along with Eisners algorithm, to yield the expected improvement.
---------------------------------------------------
P06-2009:10	192:229	4.2 Evaluation We use the same evaluation metrics as in (McDonald et al. , 2005).
---------------------------------------------------
P06-2009:11	196:229	When evaluating the result, we exclude the punctuation marks, as done in (McDonald et al. , 2005) and (Yamada and Matsumoto, 2003).
---------------------------------------------------
P06-2009:12	33:229	Dependency structures are more ef cient to parse (Eisner, 1996) and are believed to be easier to learn, yet they still capture much of the predicate-argument information needed in applications (Haghighi et al. , 2005), which is one reason for the recent interest in learning these structures (Eisner, 1996; McDonald et al. , 2005; Yamada and Matsumoto, 2003; Nivre and Scholz, 2004).
---------------------------------------------------
D07-1126:13	90:136	Table 5 reports experimental results by using the first order decoding method in which an MST parsing algorithm (McDonald et al. , 2005b) is applied for non-projective parsing and the Eisners method is used for projective language data.
---------------------------------------------------
D07-1126:14	78:136	This feature was particularly helpful for nouns identifying their parent (McDonald et al. , 2005a).
---------------------------------------------------
D07-1126:15	14:136	The performance of MIRA based parsing achieves the state-ofthe-art performance in English data (McDonald et al. , 2005a; McDonald et al. , 2006).
---------------------------------------------------
D07-1126:16	7:136	1 Introduction Research on dependency parsing is mainly based on machine learning methods, which can be called history-based (Yamada and Matsumoto, 2003; Nivre et al. , 2006) and discriminative learning methods (McDonald et al. , 2005a; Corston-Oliver et al. , 2006).
---------------------------------------------------
D07-1126:17	43:136	For dependency parsing domain, McDonald et al (2005a) modified the MIRA learning algorithm (McDonald et al. , 2005a) for structured domains in which the optimization problem can be solved by using Hidreths algorithm (Censor and Zenios, 1997), which is faster than the quadratic programming technique.
---------------------------------------------------
D07-1126:18	36:136	The algorithms are commonly used in other online-learning dependency parsing, such as in (McDonald et al. , 2005a).
---------------------------------------------------
D07-1126:19	58:136	It is easy to see that the main difference between the PA algorithms and the Perceptron algorithm (PC) (Collins, 2002) as well as the MIRA algorithm (McDonald et al. , 2005a) is in line 9.
---------------------------------------------------
D07-1126:20	57:136	Averaging has been shown to help reduce overfitting (McDonald et al. , 2005a; Collins, 2002).
---------------------------------------------------
W06-2925:21	9:112	The features used to score, while based on the previous work in dependency parsing (McDonald et al. , 2005), introduce some novel concepts such as better codification of context and surface distances, and runtime information from dependencies previously parsed.
---------------------------------------------------
W06-2925:22	53:112	Most of these features are inspired by previous work in dependency parsing (McDonald et al. , 2005; Collins, 1999).
---------------------------------------------------
W06-2925:23	36:112	partial parsing (Carreras et al. , 2005) or even dependency parsing (McDonald et al. , 2005).
---------------------------------------------------
P06-2041:24	8:166	Alternatively, discriminative models can be used to search the complete space of possible parses (Taskar et al. , 2004; McDonald et al. , 2005).
---------------------------------------------------
W09-1215:25	9:139	Graph-based algorithms (Eisner, 1996; McDonald et al., 2005) assume a series of dependency tree candidates for a sentence and the goal is to find the dependency tree with highest score.
---------------------------------------------------
D07-1100:26	36:206	The best projective parse tree is obtained using the Eisner algorithm (Eisner, 1996) with the scores, and the best non-projective one is obtained using the ChuLiu-Edmonds (CLE) algorithm (McDonald et al. , 2005b).
---------------------------------------------------
N09-3011:27	32:108	The training algorithm we propose in this paper is based on the K-best MIRA algorithm which has been used earlier in structured prediction problems (McDonald et al., 2005a; McDonald et al., 2005b).
---------------------------------------------------
W06-1615:28	147:260	MIRA has been used successfully for both sequence analysis (McDonald et al. , 2005a) and dependency parsing (McDonald et al. , 2005b).
---------------------------------------------------
D07-1003:29	171:243	The resulting POS-tagged sentences were then parsed using MSTParser (McDonald et al. , 2005), trained on the entire Penn Treebank to produce labeled dependency parse trees (we used a coarse dependency label set that includes twelve label types).
---------------------------------------------------
D07-1003:30	97:243	The tree is produced by a state-of-the-art dependency parser (McDonald et al. , 2005) trained on the Wall Street Journal Penn Treebank (Marcus et al. , 1993).
---------------------------------------------------
P09-1087:31	98:217	This modification is essential inordertomakeourparserrunintrueO(n2)time,asopposed to (McDonald et al., 2005b).
---------------------------------------------------
P09-1087:32	207:217	6 Conclusion and future work In this paper, we presented a non-projective dependency parser whose time-complexity of O(n2) improves upon the cubic time implementation of (McDonald et al., 2005b), and does so with little loss in dependency accuracy (.25% to .34%).
---------------------------------------------------
P09-1087:33	56:217	2.1 O(n2)-time dependency parsing for MT We now formalize weighted non-projective dependency parsing similarly to (McDonald et al., 2005b) and then describe a modified and more efficient version that can be integrated into a phrasebased decoder.
---------------------------------------------------
P09-1087:34	103:217	Since i and j are both free variables, feature computation in (McDonald et al., 2005b) takes time O(n3), even though parsing itself takes O(n2) time.
---------------------------------------------------
P09-1087:35	59:217	We score each dependency relation using a standard linear model s(i, j) = f(i, j) (1) whose weight vector  is trained using MIRA (Crammer and Singer, 2003) to optimize dependency parsing accuracy (McDonald et al., 2005a).
---------------------------------------------------
P09-1087:36	109:217	The O(n3) non-projective parser of (McDonald et al., 2005b) is slightly more accurate than our version, though ours runs in O(n2) time.
---------------------------------------------------
P09-1087:37	29:217	2 Dependency parsing for machine translation In this section, we review dependency parsing formulated as a maximum spanning tree problem (McDonald et al., 2005b), which can be solved in quadratic time, and then present its adaptation and novel application to phrase-based decoding.
---------------------------------------------------
P09-1087:38	55:217	In the caseofdependencyparsingforCzech,(McDonald et al., 2005b) even outperforms projective parsing, and was one of the top systems in the CoNLL-06 shared task in multilingual dependency parsing.
---------------------------------------------------
P09-1087:39	54:217	Thatisnotthecase: dependencyaccuracyfornonprojective parsing is 90.2% for English (McDonald et al., 2005b), only 0.7% lower than a projective parser (McDonald et al., 2005a) that uses the same set of features and learning algorithm.
---------------------------------------------------
P09-1087:40	147:217	The training data consists of about 28 million English words and 23.3 million 5Note that our results on WSJ are not exactly the same as those reported in (McDonald et al., 2005b), since we used slightly different head finding rules.
---------------------------------------------------
P09-1087:41	137:217	Table 4 shows that the accuracy of our truly 777 O(n2) parser is only .25% to .34% worse than the O(n3) implementation of (McDonald et al., 2005b).5 Compared to the state-of-the-art projective parser as implemented in (McDonald et al., 2005a), performance is 1.28% lower on WSJ, but only 0.95% when training on all our available data and using the MT setting.
---------------------------------------------------
N09-1069:42	181:198	One sees this clear trend in the supervised NLP literature examples include the Perceptron algorithm for tagging (Collins, 2002), MIRA for dependency parsing (McDonald et al., 2005), exponentiated gradient algorithms (Collins et al., 2008), stochastic gradient for constituency parsing (Finkel et al., 2008), just to name a few.
---------------------------------------------------
P08-1108:43	79:172	More precisely, dependency arcs (or pairs of arcs) are first represented by a high dimensional feature vector f(i,j,l)  Rk, where f is typically a binary feature vector over properties of the arc as well as the surrounding input (McDonald et al., 2005a; McDonald et al., 2006).
---------------------------------------------------
P08-1108:44	33:172	As a result, the dependency parsing problem is written: G = argmax G=(V,A) summationdisplay (i,j,l)A s(i,j,l) This problem is equivalent to finding the highest scoring directed spanning tree in the complete graph over the input sentence, which can be solved in O(n2) time (McDonald et al., 2005b).
---------------------------------------------------
P08-1108:45	35:172	An advantage of graph-based methods is that tractable inference enables the use of standard structured learning techniques that globally set parameters to maximize parsing performance on the training set (McDonald et al., 2005a).
---------------------------------------------------
E06-1010:46	21:122	Moreover, the study of formal grammarsisonly partially relevant for research ondatadriven dependency parsing, where most systems are not grammar-based but rely on inductive inference from treebank data (Yamada and Matsumoto, 2003; Nivre et al. , 2004; McDonald et al. , 2005a).
---------------------------------------------------
N06-1021:47	147:189	From the fact that neither MIRA nor BPM clearly outperforms the other, we conclude that we have successfully replicated the results reported in (McDonald et al. , 2005a) for English.
---------------------------------------------------
N06-1021:48	145:189	For the comparison of English dependency accuracy excluding punctuation, MIRA and BPM are both statistically significantly better than the averaged perceptron result reported in (McDonald et al. , 2005a).
---------------------------------------------------
N06-1021:49	141:189	The results reported here for English and Czech are comparable to the previous best published numbers in (McDonald et al. , 2005a), as Table 3 shows.
---------------------------------------------------
N06-1021:50	67:189	2 Refer to (McDonald et al. , 2005b) for a detailed treatment of both algorithms.
---------------------------------------------------
N06-1021:51	151:189	Again, since neither MIRA nor BPM outperforms the other on all measures, we conclude that the results constitute a valiation of the results reported in (McDonald et al. , 2005a).
---------------------------------------------------
N06-1021:52	80:189	Examples include the margin perceptron (Duda et al. , 2001), ALMA (Gentile, 2001), and MIRA (which is used to train the parser in (McDonald et al. , 2005a)).
---------------------------------------------------
N06-1021:53	102:189	The feature types are essentially those described in (McDonald et al. , 2005a).
---------------------------------------------------
N06-1021:54	54:189	1 3 Parser Architecture We take as our starting point a re-implementation of McDonalds state-of-the-art dependency parser (McDonald et al. , 2005a).
---------------------------------------------------
N06-1021:55	183:189	7 Conclusions We have successfully replicated the state-of-the-art results for dependency parsing (McDonald et al. , 2005a) for both Czech and English, using Bayes Point Machines.
---------------------------------------------------
N06-1021:56	165:189	Perceptron 82.9 88.0 30.3 (inc punc) MIRA 83.3 88.6 31.3 Bayes Point Machine 84.0 88.8 30.9 Table 3: Comparison to previous best published results reported in (McDonald et al. , 2005a).
---------------------------------------------------
W06-2904:57	51:160	For standard scoring functions, parsing requires an a72a58a4a6a73a75a74a12a17 dynamic programming algorithm to compute a projective tree that obtains the maximum score (Eisner and Satta, 1999; Wang et al. , 2005; McDonald et al. , 2005).
---------------------------------------------------
W06-2904:58	8:160	1 Introduction Over the past decade, there has been tremendous progress on learning parsing models from treebank data (Collins, 1997; Charniak, 2000; Wang et al. , 2005; McDonald et al. , 2005).
---------------------------------------------------
W06-2904:59	67:160	To train a77 we follow the large margin training approach of (Taskar et al. , 2003; Tsochantaridis et al. , 2004), which has been applied with great success to dependency parsing (Taskar et al. , 2004; McDonald et al. , 2005).
---------------------------------------------------
W06-2904:60	68:160	Large margin training can be expressed as minimizing a regularized loss (Hastie et al. , 2004) a45a58a113a115a114 a77 a116a117 a77 a78 a77 a118 (3) a59 a20 a45a47a39a26a48 a119a67a62a121a120 a4a123a122a25a20a124a9 a19 a20a100a17a102a101a125a4 sa4a68a77a75a9 a19 a20a100a17a126a101 sa4a68a77a75a9a44a122a25a20a100a17a14a17 where a19 a20 is the target tree for sentence a2 a20 ; a122 a20 ranges over all possible alternative trees in a31a32a4a33a2a127a20a33a17 ; sa4a68a77a102a9a19 a17a128a3 a129 a53a61a60a63a62a95a64a32a60a67a66a130a57a68a50a29a49 a77a54a78a102a79a67a4a6a5a16a20a105a21 a5a25a24a71a17 ; and a120 a4a123a122a25a20a124a9 a19 a20a100a17 is a measure of distance between the two trees a122a25a20 and a19 a20 . Using the techniques of (Hastie et al. , 2004) one can show that minimizing (4) is equivalent to solving the quadratic program a45a47a113a115a114 a131a71a132a133 a116a117 a77a54a78a102a77a134a118a98a135a136a78a75a137 subject to (4) a138 a20a102a139 a120 a4 a19 a20a97a9a44a122a140a20a100a17a103a118 sa4a68a77a75a9a44a122a25a20a68a17a126a101 sa4a68a77a102a9 a19 a20a33a17 for all a141a44a9a44a122a25a20 a35 a31a32a4a33a2a142a20a68a17 which corresponds to the training problem posed in (McDonald et al. , 2005).
---------------------------------------------------
W06-2904:61	13:160	Currently, the work on conditional parsing models appears to have culminated in large margin training (Taskar et al. , 2003; Taskar et al. , 2004; Tsochantaridis et al. , 2004; McDonald et al. , 2005), which currently demonstrates the state of the art performance in English dependency parsing (McDonald et al. , 2005).
---------------------------------------------------
W06-2904:62	72:160	Although (McDonald et al. , 2005) explicitly describes this as an advantage over previous approaches (Ratnaparkhi, 1999; Yamada and Matsumoto, 2003), below we nd that changing the loss to enforce a more detailed set of constraints leads to a more effective approach.
---------------------------------------------------
W06-2904:63	18:160	It turns out that probabilistic approaches pay closer attention to the individual errors made by each component of a parse, whereas the training error minimized in the large margin approach the structured margin loss (Taskar et al. , 2003; Tsochantaridis et al. , 2004; McDonald et al. , 2005) is a coarse measure that only assesses the total error of an entire parse rather than focusing on the error of any particular component.
---------------------------------------------------
W06-2904:64	14:160	Despite the realization that maximum margin training is closely related to maximum conditional likelihood for conditional models (McDonald et al. , 2005), a suf ciently uni ed view has not yet been achieved that permits the easy exchange of improvements between the probabilistic and nonprobabilistic approaches.
---------------------------------------------------
W06-2904:65	70:160	First, there are exponentially many constraints corresponding to each possible parse of each training sentence which forces one to use alternative training procedures, such as incremental constraint generation, to slowly converge to a solution (McDonald et al. , 2005; Tsochantaridis et al. , 2004).
---------------------------------------------------
W06-2904:66	50:160	In which case, the parsing problem reduces to a19a38a37 a3a40a39a42a41a44a43a46a45a47a39a26a48 a49a51a50a42a52a54a53a56a55a58a57 a59 a53a61a60a63a62a6a64a65a60a67a66a44a57a68a50a29a49 sa4a6a5 a20 a21a69a5 a24 a17 (1) where the score sa4a6a5a27a20a70a21 a5a25a24a71a17 can depend on any measurable property of a5a30a20 and a5a25a24 within the tree a19 . This formulation is suf ciently general to capture most dependency parsing models, including probabilistic dependency models (Wang et al. , 2005; Eisner, 1996) as well as non-probabilistic models (McDonald et al. , 2005).
---------------------------------------------------
W06-2904:67	49:160	Dependency trees are usually assumed to be projective (no crossing arcs), which means that if there is an arc a4a6a5a27a20a25a21a28a5a25a24a29a17, then a5a16a20 is an ancestor of all the words between a5a30a20 and a5a25a24 . Let a31a32a4a33a2a34a17 denote the set of all the directed, projective trees that span a2 . Given an input sentence a2, we would like to be able to compute the best parse; that is, a projective tree, a19a36a35 a31a32a4a33a2a34a17, that obtains the highest score . In particular, we follow (Eisner, 1996; Eisner and Satta, 1999; McDonald et al. , 2005) and assume that the score of a complete spanning tree a19 for a given sentence, whether probabilistically motivated or not, can be decomposed as a sum of local scores for each link (a word pair).
---------------------------------------------------
P07-1050:68	6:193	1 Introduction The Maximum Spanning Tree algorithm1 was recently introduced as a viable solution for nonprojective dependency parsing (McDonald et al. , 2005b).
---------------------------------------------------
D08-1052:69	120:235	Penn Treebank was previously used to train and evaluate various dependency parsers (Yamada and Matsumoto, 2003; McDonald et al., 2005).
---------------------------------------------------
D08-1052:70	209:235	23 of the LTAG Treebank dency is comparable to the numbers of the previous best systems on dependency extracted from PTB with Magermans rules, for example, 90.3% in (Yamada and Matsumoto, 2003) and 90.9% in (McDonald et al., 2005).
---------------------------------------------------
N06-3004:71	9:68	This is also true for reranking and discriminative training, where the k-best list of candidates serves as an approximation of the full set (Collins, 2000; Och, 2003; McDonald et al. , 2005).
---------------------------------------------------
N06-3004:72	20:68	These algorithms have been re-implemented by other researchers in the field, including Eugene Charniak for his n-best parser, Ryan McDonald for his dependency parser (McDonald et al. , 2005), Microsoft Research NLP group (Simon Corston-Oliver and Kevin Duh, p.c).
---------------------------------------------------
P09-2026:73	26:88	The system uses as input the paired corpus, the corresponding POS tagged corpus, the paired corpus parsed using the Charniak parser (Charniak, 2000), and dependency parses from the MST parser (McDonald et al., 2005).
---------------------------------------------------
P06-1091:74	26:210	Moreover, under this view, SMT becomes quite similar to sequential natural language annotation problems such as part-of-speech tagging and shallow parsing, and the novel training algorithm presented in this paper is actually most similar to work on training algorithms presented for these task, e.g. the on-line training algorithm presented in (McDonald et al. , 2005) and the perceptron training algorithm presented in (Collins, 2002).
---------------------------------------------------
D07-1013:75	40:290	Ax = {(i,j,l) | i,j  Vx and l  L} Let D(Gx) represent the subgraphs of graph Gx that are valid dependency graphs for the sentence x. Since Gx contains all possible labeled arcs, the set D(Gx) must necessarily contain all valid dependency graphs for x. Assume that there exists a dependency arc scoring function, s : V V L  R. Furthermore, define the score of a graph as the sum of its arc scores, s(G = (V,A)) = summationdisplay (i,j,l)A s(i,j,l) The score of a dependency arc, s(i,j,l) represents the likelihood of creating a dependency from word wi to word wj with the label l. If the arc score function is known a priori, then the parsing problem can be stated as, 123 G = argmax GD(Gx) s(G) = argmax GD(Gx) summationdisplay (i,j,l)A s(i,j,l) This problem is equivalent to finding the highest scoring directed spanning tree in the graph Gx originatingoutoftherootnode0, whichcanbesolvedfor both the labeled and unlabeled case in O(n2) time (McDonald et al. , 2005b).
---------------------------------------------------
D07-1013:76	43:290	To learn arc scores, these models use large-margin structured learning algorithms (McDonald et al. , 2005a), which optimize the parameters of the model to maximize the score margin between the correct dependency graph and all incorrect dependency graphs for every sentence in a training set.
---------------------------------------------------
W06-1616:77	81:238	However, global constraints cannot be incorporated into the CLE algorithm (McDonald et al. , 2005b).
---------------------------------------------------
W06-1616:78	117:238	The constraints are chosen based on the two criteria: (1) adding them to the base constraints (those added in advance) would result in an extremely large program, and (2) it must be efcient to detect whether the constraint is violated in y. No Cycles (T2) For every possible cycle c for the sentence x we have a constraint which forbids the case where all edges in c are active simultaneously: summationdisplay (i,j)c di,j |c|1 Comma Coordination (C3) For each symmetric conjunction token i which forms a symmetric coordination and each set of tokens A in x to the left of i with no comma between each pair of successive tokens we add: summationdisplay aA di,a |A|1 which forbids con gurations where i has the argument tokens A. Compatible Coordination Arguments (C4) For each conjunction token i and each set of tokens A in x with incompatible POS tags, we add a constraint to forbid con gurations where i has the argument tokens A. summationdisplay aA di,a |A|1 Selective Projective Parsing (P1) For each pair of triplets (i,j,l1) and (m,n,l2) we add the constraint: ei,j,l1 + em,n,l2 1 if l1 or l2 is in P. 3.2 Training For training we use single-best MIRA (McDonald et al. , 2005a).
---------------------------------------------------
W06-1616:79	47:238	The verb krijg is incorrectly coordinated with the preposition om . work is ef cient and has also been extended to non-projective trees (McDonald et al. , 2005b).
---------------------------------------------------
W06-1616:80	188:238	While we expect a longer runtime than using the Chu-Liu-Edmonds as in previous work (McDonald et al. , 2005b), we are interested in how large the increase is. Table 2 shows the average solve time (ST) for sentences with respect to the number of tokens in each sentence for our system with constraints (cnstr) and the Chu-Liu-Edmonds (CLE) algorithm.
---------------------------------------------------
W06-1616:81	8:238	Strong assumptions are also made in the case of McDonald et al.s (2005b) non-projective dependency parsing model.
---------------------------------------------------
W06-1616:82	234:238	This allows us to ef ciently use ILP for dependency parsing and add constraints which provide a signi cant improvement over the current stateof-the-art parser (McDonald et al. , 2005b) on the Dutch Alpino corpus (see bl row in Table 1).
---------------------------------------------------
C08-1050:83	52:214	A dependency-based system using MSTParser (McDonald et al., 2005).
---------------------------------------------------
C08-1050:84	39:214	The head rules created by Yamada and Matsumoto (2003) have been used in almost all recent work on statistical dependency parsing of English (Nivre and Scholz, 2004; McDonald et al., 2005).
---------------------------------------------------
D08-1017:85	10:217	1, on which we focus in this paper:  Exact algorithms for dependency parsing (Eisner and Satta, 1999; McDonald et al., 2005b) are tractable only when the model makes very strong, linguistically unsupportable independence assumptions, such as arc factorization for nonprojective dependency parsing (McDonald and Satta, 2007).
---------------------------------------------------
D08-1017:86	20:217	Such models are commonlyreferredto as edge-factored since their parametersfactor relative to individual edges of the graph (Paskin, 2001; McDonald et al., 2005a).Edge-factoredmodelshave many computationalbenefits,mostnotablythat inferencefor nonprojective dependency graphs can be achieved in polynomialtime(McDonaldet al., 2005b).Theprimary problem in treating each dependency as independentis that it is not a realistic assumption.
---------------------------------------------------
D08-1017:87	47:217	1.1 RelatedWork There has been extensive work on data-driven dependency parsingfor both projective parsing(Eisner, 1996; Paskin, 2001; Yamadaand Matsumoto, 2003; Nivre and Scholz, 2004; McDonaldet al., 2005a) and non-projective parsingsystems(Nivre and Nilsson,2005;Halland Novak, 2005;McDonald et al., 2005b).
---------------------------------------------------
D08-1017:88	72:217	In graph-based parsing, dependency trees are scored by factoring the tree into its arcs, and parsing is performed by searching for the highest scoring tree (Eisner, 1996; McDonald et al., 2005b).
---------------------------------------------------
D08-1017:89	3:217	A typical approach in graph-based dependency parsing has been to assume a factorized model, where local features are used but a global function is optimized (McDonald et al., 2005b).
---------------------------------------------------
D08-1017:90	38:217	Such models are commonlyreferredto as edge-factored since their parametersfactor relative to individualedges of the graph (Paskin, 2001; McDonald et al., 2005a).Edge-factoredmodelshave many computationalbenefits,mostnotablythatinferencefor nonprojective dependency graphs can be achieved in polynomialtime(McDonaldet al.,2005b).Theprimary problemin treating each dependency as independentis that it is not a realisticassumption.
---------------------------------------------------
D08-1017:91	106:217	In this work we use two well-known, publicly available dependency parsers, MSTParser (McDonald et al., 2005b),1 which implements ex1http://sourceforge.net/projects/mstparser 159 act first-order arc-factored nonprojective parsing (2.1.2) and approximate second-order nonprojective parsing, and MaltParser (Nivre et al., 2006), which is a state-of-the-art transition-based parser.2 We do not alter the training algorithms used in prior work for learning these two parsers from data.
---------------------------------------------------
D08-1017:92	22:217	However, in the data-driven parsingsetting this can be partiallyadvertedby incorporatingrich featurerepresentationsovertheinput(McDonaldet al.,2005a).
---------------------------------------------------
D08-1017:93	15:217	Specifically, we view stacked learning as a way of approximating non-local features in a linear model, rather than making empirically dubious independence (McDonald et al., 2005b) or structural assumptions (e.g., projectivity, Eisner, 1996), using search approximations (Sagae and Lavie, 2005; Hall et al., 2006; McDonald and Pereira, 2006), solving a (generally NP-hard) integer linear program (Riedel and Clarke, 2006), or adding latent variables (Titov and Henderson, 2007).
---------------------------------------------------
D08-1017:94	29:217	1.1 RelatedWork There has been extensive work on data-driven dependency parsingfor both projective parsing(Eisner, 1996; Paskin, 2001; Yamada and Matsumoto, 2003; Nivre and Scholz, 2004; McDonaldet al., 2005a) and non-projective parsing systems (Nivre and Nilsson,2005;Hall and Novak, 2005;McDonald et al., 2005b).
---------------------------------------------------
D08-1017:95	32:217	In the second category are those that employ exhaustive inferencealgorithms,usuallybymakingstrongindependenceassumptions,as is the case for edge-factoredmodels(Paskin,2001; McDonaldet al., 2005a; McDonaldet al., 2005b).
---------------------------------------------------
D08-1017:96	150:217	We adapted this system to first perform unlabeled parsing, then label the arcs using a log-linear classifier with access to the full unlabeled parse (McDonald et al., 2005a; McDonald et al., 2005b; McDonald and Pereira, 2006).
---------------------------------------------------
D08-1017:97	49:217	In the second category are those that employ exhaustive inferencealgorithms,usuallybymakingstrongindependenceassumptions,as is the casefor edge-factoredmodels(Paskin,2001; McDonaldet al., 2005a;McDonaldet al., 2005b).
---------------------------------------------------
W07-1509:98	19:105	Our experiments use the MIRA algorithm (Crammer et al. , 2006; McDonald et al. , 2005) to learn the weight vector.
---------------------------------------------------
W07-1509:99	26:105	We used a k53 best version of the MIRA algorithm (Crammer et al. , 2006; McDonald et al. , 2005).
---------------------------------------------------
D09-1085:100	101:198	Also, dependency parsers are not significantly better at recovering head-based dependencies than constituent parsers based on the PTB (McDonald et al., 2005).
---------------------------------------------------
D09-1085:101	99:198	The RASP parser is based on a manually constructed POS tag-sequence grammar, with a statistical parse selection component and a robust 1One obvious omission is any form of dependency parser (McDonald et al., 2005; Nivre and Scholz, 2004).
---------------------------------------------------
P07-2052:102	45:109	where S is the stack that keeps the words being under consideration, I is the list of reDA RA CR (Yamada and Matsumoto, 2003) 90.3 91.6 38.4 (Nivre and Scholz, 2004) 87.3 84.3 30.4 (Isozaki et al. , 2004) 91.2 95.7 40.7 (McDonald et al. , 2005) 90.9 94.2 37.5 (McDonald and Pereira, 2006) 91.5 N/A 42.1 (Corston-Oliver et al. , 2006) 90.8 93.7 37.6 Our Base Parser 90.9 92.6 39.2 Table 2: Comparison of parser performance.
---------------------------------------------------
P07-2052:103	35:109	McDonald et al. proposed an online large-margin method for training dependency parsers (McDonald et al. , 2005).
---------------------------------------------------
P07-2052:104	33:109	Dependency parsing has been actively studied in recent years (Yamada and Matsumoto, 2003; Nivre and Scholz, 2004; Isozaki et al. , 2004; McDonald et al. , 2005; McDonald and Pereira, 2006; Corston-Oliver et al. , 2006).
---------------------------------------------------
D07-1070:105	4:458	We argue that bootstrapping a parser is most promising when the model uses a rich set of redundant features, as in recent models for scoring dependency parses (McDonald et al. , 2005).
---------------------------------------------------
D08-1059:106	30:188	2 The graph-based parser Following MSTParser (McDonald et al., 2005; McDonald and Pereira, 2006), we define the graphVariables: agenda  the beam for state items item  partial parse tree output  a set of output items index,prev  word indexes Input: x  POS-tagged input sentence.
---------------------------------------------------
D08-1059:107	7:188	1 Introduction Graph-based (McDonald et al., 2005; McDonald and Pereira, 2006; Carreras et al., 2006) and transition-based (Yamada and Matsumoto, 2003; Nivre et al., 2006) parsing algorithms offer two different approaches to data-driven dependency parsing.
---------------------------------------------------
D08-1059:108	34:188	We use the discriminative perceptron learning algorithm (Collins, 2002; McDonald et al., 2005) to train the values of vectorw.
---------------------------------------------------
D08-1059:109	130:188	Rows MSTParser 1/2 show the first-order (using feature templates 1  5 from Table 1) (McDonald et al., 2005) and secondorder (using all feature templates from Table 1) (McDonald and Pereira, 2006) MSTParsers, as reported by the corresponding papers.
---------------------------------------------------
J08-4003:110	560:595	This approach has been further developed in particular by Ryan McDonald and his colleagues (McDonald, Crammer, and Pereira 2005; McDonald et al. 2005; McDonald and Pereira 2006) and is now known as spanning tree parsing, because the problem of nding the most probable tree under this type of model is equivalent to nding an optimum spanning tree in a dense graph containing all possible dependency arcs.
---------------------------------------------------
W08-2131:111	18:136	The parameters were determined based on the experimental results of the English task in (McDonald et al., 2005), i.e. we used projective parsing and a first order feature set during training.
---------------------------------------------------
P09-1093:112	87:165	For example,  w is optimized as follows:  w(new) =  w(old)  epsilon1 L  w(old) (11) 829 Our parameter optimization procedure can be replaced by another one such as MIRA (McDonald et al., 2005) or CRFs (Lafferty et al., 2001).
---------------------------------------------------
P09-1040:113	7:154	Thus, transition-based parsers normally run in linear or quadratic time, using greedy deterministic search or fixed-width beam search (Nivre et al., 2004; Attardi, 2006; Johansson and Nugues, 2007; Titov and Henderson, 2007), and graph-based models support exact inference in at most cubic time, which is efficient enough to make global discriminative training practically feasible (McDonald et al., 2005a; McDonald et al., 2005b).
---------------------------------------------------
P09-1040:114	18:154	Either they employ a non-standard parsing algorithm that can combine non-adjacent substructures (McDonald et al., 2005b; Attardi, 2006; Nivre, 2007), or they try to recover non351 ROOT0 A1 a7 a4 a63 DET hearing2 a7 a4 a63 SBJ is3 a7 a4 a63 ROOT scheduled4 a7 a4 a63 VG on5 a7 a4 a63 NMOD the6 a7 a4 a63 DET issue7 a7 a4 a63 PC today8 a7 a4 a63 ADV .9 a63 a7 a4P Figure 1: Dependency tree for an English sentence (non-projective).
---------------------------------------------------
P09-1040:115	5:154	1 Introduction Syntactic parsing using dependency structures has become a standard technique in natural language processing with many different parsing models, in particular data-driven models that can be trained on syntactically annotated corpora (Yamada and Matsumoto, 2003; Nivre et al., 2004; McDonald et al., 2005a; Attardi, 2006; Titov and Henderson, 2007).
---------------------------------------------------
P08-1067:116	16:181	Alternatively, discriminative parsing is tractable with exact and efficient search based on dynamic programming (DP) if all features are restricted to be local, that is, only looking at a local window within the factored search space (Taskar et al., 2004; McDonald et al., 2005).
---------------------------------------------------
D09-1021:117	67:224	In particular, we will consider discriminative models (analogous to models for dependency parsing, e.g., see (McDonald et al., 2005)) that estimate the probability of targetlanguage dependencies conditioned on properties of the source-language string.
---------------------------------------------------
D09-1021:118	20:224	Inspired by work in discriminative dependency parsing (e.g., (McDonald et al., 2005)), we add probabilistic constraints to the model through a discriminative model that links lexical dependencies in the target language to features of the source language string.
---------------------------------------------------
P09-1059:119	82:154	This is similar to feature design in discriminative dependency parsing (McDonald et al., 2005; Mc525 Donald and Pereira, 2006), where the basic features, composed of words and POSs in the context, are also conjoined with link direction and distance in order to obtain more special features.
---------------------------------------------------
P09-1041:120	152:264	The second full set includes standard features for edgefactored dependency parsers (McDonald et al., 2005), though still unlexicalized.
---------------------------------------------------
P09-1041:121	186:264	We also use standard edge-factored feature templates (McDonald et al., 2005)10.
---------------------------------------------------
P09-1041:122	204:264	In some cases, the possible parent constraints described above will not be enough to provide high accuracy, because they do not consider other tags in the sentence (McDonald et al., 2005).
---------------------------------------------------
W06-2928:123	19:104	As described in (Corston-Oliver et al. , 2006), we reimplemented the parser described in (McDonald et al. , 2005) and validated their results for Czech and English.
---------------------------------------------------
W06-2928:124	28:104	Additional features are created by combining these atomic features, as described in (McDonald et al. , 2005).
---------------------------------------------------
P07-1055:125	36:215	These algorithms are usually applied to sequential labeling or chunking, but have also been applied to parsing (Taskar et al. , 2004; McDonald et al. , 2005), machine translation (Liang et al. , 2006) and summarization (Daume III et al. , 2006).
---------------------------------------------------
P07-1055:126	35:215	Hidden Markov models (Rabiner, 1989) are one of the earliest structured learning algorithms, which have recently been followedbydiscriminativelearningapproachessuch as conditional random fields (CRFs) (Lafferty et al. , 2001; Sutton and McCallum, 2006), the structured perceptron (Collins, 2002) and its large-margin variants (Taskar et al. , 2003; Tsochantaridis et al. , 2004; McDonald et al. , 2005; Daume III et al. , 2006).
---------------------------------------------------
P07-1055:127	105:215	2.1.3 Training the Model Let Y = Y(d)  Y(s)n be the set of all valid sentence-document labelings for an input s. The weights, w, are set using the MIRA learning algorithm, which is an inference based online largemargin learning technique (Crammer and Singer, 2003; McDonald et al. , 2005).
---------------------------------------------------
P07-1055:128	119:215	The constraint set C can be chosen arbitrarily, but it is usually taken to be the k labelings that have the highest score under the old weight vector w(i) (McDonald et al. , 2005).
---------------------------------------------------
W05-1506:129	58:254	Although generative models, being probabilitybased, do not suffer from this problem, more general models (e.g. , log-linear models) may require negative edge costs (McDonald et al. , 2005; Taskar et al. , 2004).
---------------------------------------------------
P09-1043:130	52:231	This not to say that domain adaptation is 379 HPSG DBExtraction HPSG DBFeature Models MSTParserFeature Model MaltParserFeature Model Section 3.1 Section 3.3 McDonaldet al., 2005 Nivreet al., 2007 Nivre andMcDonald, 2008 Section 4.2 Section 4.3 Figure 1: Different dependency parsing models and their combinations.
---------------------------------------------------
P09-1043:131	4:231	In combination with machine learning methods, several statistical dependency parsing models have reached comparable high parsing accuracy (McDonald et al., 2005b; Nivre et al., 2007b).
---------------------------------------------------
P09-1043:132	18:231	2 Parser Domain Adaptation In recent years, two statistical dependency parsing systems, MaltParser (Nivre et al., 2007b) and MSTParser (McDonald et al., 2005b), representing different threads of research in data-driven machine learning approaches have obtained high publicity, for their state-of-the-art performances in open competitions such as CoNLL Shared Tasks.
---------------------------------------------------
P09-1043:133	103:231	Therefore, the feature model focuses on each kind of head-child pair in the dependency tree, and mainly contains four categories of features (Mcdonald et al., 2005a): basic uni-gram features, basic bi-gram features, in-between POS features, and surrounding POS features.
---------------------------------------------------
P09-1043:134	20:231	MSTParser, on the other hand, follows 378 the graph-based approach where the best parse tree is acquired by searching for a spanning tree which maximizes the score on either a partially or a fully connected graph with all words in the sentence as nodes (Eisner, 1996; McDonald et al., 2005b).
---------------------------------------------------
D08-1016:135	91:363	A first-order (or edge-factored) parsing model (McDonald et al., 2005) contains only LINK factors, along with a global TREE or PTREE factor.
---------------------------------------------------
W06-2932:136	12:130	This has been shown through their successful use in many standard natural language processing tasks, including machine translation (Ding and Palmer, 2005), sentence compression (McDonald, 2006), and textual inference (Haghighi et al. , 2005).
---------------------------------------------------
W06-2932:137	71:130	These results show that the discriminative spanning tree parsing framework (McDonald et al. , 2005b; McDonald and Pereira, 2006) is easily adapted across all these languages.
---------------------------------------------------
W06-2932:138	44:130	We use the MIRA 217 online learner to set the weights (Crammer and Singer, 2003; McDonald et al. , 2005a) since we found it trained quickly and provide good performance.
---------------------------------------------------
W06-2932:139	124:130	(McDonald et al. , 2005b; McDonald and Pereira, 2006) generalizes well to languages other than English.
---------------------------------------------------
W06-2932:140	21:130	That work extends the maximum spanning tree dependency parsing framework (McDonald et al. , 2005a; McDonald et al. , 2005b) to incorporate features over multiple edges in the dependency graph.
---------------------------------------------------
W06-2920:141	47:416	(2005), while McDonalds parser has been applied to English (McDonald et al. , 2005a), Czech (McDonald et al. , 2005b) and, very recently, Danish (McDonald and Pereira, 2006).
---------------------------------------------------
W06-2920:142	236:416	The search for the best parse can then be formalized as the search for the maximum spanning tree (MST) (McDonald et al. , 2005b).
---------------------------------------------------
P08-1061:143	6:148	1 Introduction Supervised learning algorithms still represent the state of the art approach for inferring dependency parsers from data (McDonald et al., 2005a; McDonald and Pereira, 2006; Wang et al., 2007).
---------------------------------------------------
P08-1061:144	59:148	3 Supervised Structured Large Margin Training Supervised structured large margin training approaches have been applied to parsing and produce promising results (Taskar et al., 2004; McDonald et al., 2005a; Wang et al., 2006).
---------------------------------------------------
P08-1061:145	125:148	Using the full set of features described in (McDonald et al., 2005a; Wang et al., 2007) and comparing the corresponding dependency parsing 537 English PTB-10 Training(l/ul) 3026/1016 Dev 163 Test 270 PTB-15 Training 7303/2370 Dev 421 Test 603 PTB-20 Training 12519/4003 Dev 725 Test 1034 Chinese CTB4-10 Training(l/ul) 642/347 Dev 61 Test 40 CTB4-15 Training 1262/727 Dev 112 Test 83 CTB4-20 Training 2038/1150 Dev 163 Test 118 CTB4-40 Training 4400/2452 Dev 274 Test 240 CTB4 Training 5314/2977 Dev 300 Test 289 Table 1: Size of Experimental Data (# of sentences) results with previous work remains a direction for future work.
---------------------------------------------------
P08-1061:146	123:148	Features For simplicity, in current work, we only used two sets of featuresword-pair and tag-pair indicator features, which are a subset of features used by other researchers on dependency parsing (McDonald et al., 2005a; Wang et al., 2007).
---------------------------------------------------
P08-1061:147	126:148	Dependency Parsing Algorithms For simplicity of implementation, we use a standard CKY parser in the experiments, although Eisners algorithm (Eisner, 1996) and the Spanning Tree algorithm (McDonald et al., 2005b) are also applicable.
---------------------------------------------------
P08-1061:148	56:148	In particular, one would assume that the score of a complete spanning tree Y for a given sentence, whether probabilistically motivated or not, can be decomposed as a sum of local scores for each link (a word pair) (Eisner, 1996; Eisner and Satta, 1999; McDonald et al., 2005a).
---------------------------------------------------
P08-1061:149	57:148	Given this assumption, the parsing problem reduces to find Y = arg max Y(X) score(Y|X) (1) = arg max Y(X) summationdisplay (xixj)Y score(xi  xj) where the score(xi  xj) can depend on any measurable property of xi and xj within the sentence X. This formulation is sufficiently general to capture most dependency parsing models, including probabilistic dependency models (Eisner, 1996; Wang et al., 2005) as well as non-probabilistic models (McDonald et al., 2005a).
---------------------------------------------------
P08-1061:150	65:148	This approach corresponds to the training problem posed in (McDonald et al., 2005a) and has yielded the best published results for English dependency parsing.
---------------------------------------------------
C08-5001:151	218:227	Applications of this algorithm include k-best parsing (McDonald et al., 2005; Mohri and Roark, 2006) and machine translation (Chiang, 2007).
---------------------------------------------------
C08-5001:152	200:227	The k-best list is also frequently used in discriminative learning to approximate the whole set of candidates which is usually exponentially large (Och, 2003; McDonald et al., 2005).
---------------------------------------------------
I08-2119:153	106:150	Corpus preprocessing is done as the following: sentence segmentation was performed using the tool from CCG group at UIUC 1; words are then tokenized and tagged with part-of-speech using MXPOST (Ratnaparkhi, 1996) and dependency parsing is performed using MSTParser (McDonald et al., 2005a).
---------------------------------------------------
I08-2119:154	17:150	The second type of collection that has been widely studied is biomedical literature (Bunescu and Mooney, 2005b; Giuliano et al., 2006; McDonald et al., 2005b), promoted by evaluation programs such as BioCreAtIvE and JNLPBA 2004.
---------------------------------------------------
W08-2102:155	84:209	The dependencies D must form a directed, projective tree spanning words 0n, with  at the root of this tree, as is also the case in previous work on discriminative approches to dependency parsing (McDonald et al., 2005).
---------------------------------------------------
W08-2102:156	133:209	The first-stage model we use is a first-order dependency model, with labeled dependencies, as described in (McDonald et al., 2005).
---------------------------------------------------
P08-1066:157	226:243	6 Discussion The well-formed dependency structures defined here are similar to the data structures in previous work on mono-lingual parsing (Eisner and Satta, 1999; McDonald et al., 2005).
---------------------------------------------------
D07-1062:158	48:201	We did this for three reasons: (i) our results are directly comparable to those who have used the Charniak parses distributed with the CoNLL 2005 data-set; (ii) we avoid the possibility of a better parser identifying a larger number of argument constituents and thus leading to better results, which is orthogonal to the discriminative power of our proposed LTAG-based features; and (iii) the quality of LTAG derivation trees depends indirectly on the quality of head dependencies recovered by the parser and it is a well-known folklore result (see Table 3 in (McDonald et al. , 591 2005)) that applying the head-percolation heuristics on parser output produces better dependencies when compared to dependencies directly recovered by the parser (whether the parser is an LTAG parser or a lexicalized PCFG parser).
---------------------------------------------------
D09-1058:159	118:216	as those described in (McDonald et al., 2005a; McDonald et al., 2005b; McDonald and Pereira, 2006; Koo et al., 2008).
---------------------------------------------------
D09-1058:160	33:216	In conditional log-linear models for dependency parsing (which are closely related to conditional random fields (Lafferty et al., 2001)), a distribution over dependency structures for a sentence x is defined as follows: p(y|x) = 1Z(x) exp{g(x,y)}, (1) where Z(x) is the partition function, w is a parameter vector, and g(x,y) = summationdisplay (h,m,l)y wf(x,h,m,l) Here f(x,h,m,l) is a feature vector representing the dependency (h,m,l) in the context of the sentence x (see for example (McDonald et al., 2005a)).
---------------------------------------------------
D09-1058:161	145:216	These settings match the evaluation setting in previous work such as (McDonald et al., 2005a; Koo et al., 2008).
---------------------------------------------------
D09-1058:162	132:216	Our baseline features (baseline) are very similar to those described in (McDonald et al., 2005a; Koo et al., 2008): these features track word and POS bigrams, contextual features surrounding dependencies, distance features, and so on.
---------------------------------------------------
D09-1058:163	105:216	Then, we substitute Step 3 as a supervised learning such as MIRA with a second-order parsing model (McDonald et al., 2005a), which incorporates q1 as a real-values features.
---------------------------------------------------
D09-1058:164	200:216	(a) English dependency parsers on PTB dependency parser test description (McDonald et al., 2005a) 90.9 1od (McDonald and Pereira, 2006) 91.5 2od (Koo et al., 2008) 92.23 1od, 43M ULD SS-SCM (w/ CL) 92.70 1od, 3.72G ULD (Koo et al., 2008) 93.16 2od, 43M ULD 2-stage SS-SCM(+MIRA, w/ CL) 93.79 2od, 3.72G ULD (b) Czech dependency parsers on PDT dependency parser test description (McDonald et al., 2005b) 84.4 1od (McDonald and Pereira, 2006) 85.2 2od (Koo et al., 2008) 86.07 1od, 39M ULD (Koo et al., 2008) 87.13 2od, 39M ULD SS-SCM (w/ CL) 87.14 1od, 39M ULD 2-stage SS-SCM(+MIRA, w/ CL) 88.05 2od, 39M ULD Table 6: Comparisons with the previous top systems: (1od, 2od: 1stand 2nd-order parsing model, ULD: unlabeled data).
---------------------------------------------------
P09-1042:165	37:215	To get a feel for the typical case, we used off-the-shelf parsers (McDonald et al., 2005) for English, Spanish and Bulgarian on two bitexts (Koehn, 2005; Tiedemann, 2007) and compared several measures of dependency conservation.
---------------------------------------------------
P09-1042:166	29:215	A parallel corpus is word-level aligned using an alignment toolkit (Graa et al., 2009) and the source (English) is parsed using a dependency parser (McDonald et al., 2005).
---------------------------------------------------
P09-1042:167	95:215	We used the Tokyo tagger (Tsuruoka and Tsujii, 2005) to POS tag the English tokens, and generated parses using the first-order model of McDonald et al.
---------------------------------------------------
P09-1042:168	17:215	Recently, dependency parsing has gained popularity as a simpler, computationally more efficient alternative to constituency parsing and has spurred several supervised learning approaches (Eisner, 1996; Yamada and Matsumoto, 2003a; Nivre and Nilsson, 2005; McDonald et al., 2005) as well as unsupervised induction (Klein and Manning, 2004; Smith and Eisner, 2006).
---------------------------------------------------
P09-1042:169	52:215	The discriminative parser is based on the edge-factored model and features of the MSTParser (McDonald et al., 2005).
---------------------------------------------------
P09-1007:170	110:206	Although (McDonald et al., 2005) used the prefix of each word form instead of word form itself as features, character-level features here for Chinese is essentially different from that.
---------------------------------------------------
P09-1007:171	7:206	1 Introduction Although supervised learning methods bring stateof-the-art outcome for dependency parser inferring (McDonald et al., 2005; Hall et al., 2007), a large enough data set is often required for specific parsing accuracy according to this type of methods.
---------------------------------------------------
W07-2216:172	44:259	In the second category are those that employ exhaustive inference algorithms, usually by making strong independence assumptions, as is the case for edge-factored models (Paskin, 2001; McDonald et al. , 2005a; McDonald et al. , 2005b).
---------------------------------------------------
W07-2216:173	147:259	Let R(T,Tprime) be the Hamming distance between two dependency graphs for an input sentence x = x0x1 xn, R(T,Tprime) = n  summationdisplay (i,j)kET I((i,j)k,Tprime) This is a common definition of risk between two graphs as it corresponds directly to labeled dependency parsing accuracy (McDonald et al. , 2005a; Buchholz et al. , 2006).
---------------------------------------------------
W07-2216:174	26:259	Such models are commonly referred to as edge-factored since their parameters factor relative to individual edges of the graph (Paskin, 2001; McDonald et al. , 2005a).
---------------------------------------------------
W07-2216:175	137:259	These include the perceptron (Collins, 2002) and its large-margin variants (Crammer and Singer, 2003; McDonald et al. , 2005a).
---------------------------------------------------
W07-2216:176	41:259	1.1 Related Work There has been extensive work on data-driven dependency parsing for both projective parsing (Eisner, 1996; Paskin, 2001; Yamada and Matsumoto, 2003; Nivre and Scholz, 2004; McDonald et al. , 2005a) and non-projective parsing systems (Nivre and Nilsson, 2005; Hall and Novak, 2005; McDonald et al. , 2005b).
---------------------------------------------------
W07-2216:177	27:259	Edge-factored models have many computational benefits, most notably that inference for nonprojective dependency graphs can be achieved in polynomial time (McDonald et al. , 2005b).
---------------------------------------------------
D07-1080:178	29:227	MIRA is successfully employed in dependency parsing (McDonald et al. , 2005) or the joint-labeling/chunking task (Shimizu and Haas, 2006).
---------------------------------------------------
D07-1080:179	112:227	4.1 Margin Infused Relaxed Algorithm The Margin Infused Relaxed Algorithm (MIRA) (Crammer et al. , 2006) is an online version of the large-margin training algorithm for structured classification (Taskar et al. , 2004) that has been successfully used for dependency parsing (McDonald et al. , 2005) and joint-labeling/chunking (Shimizu and Haas, 2006).
---------------------------------------------------
P08-1110:180	95:218	This is the principle behind the parser defined by Eisner (1996), which is still in wide use today (Corston-Oliver et al., 2006; McDonald et al., 2005a).
---------------------------------------------------
D07-1101:181	65:204	In turn, those features were inspired by successful previous work in firstorder dependency parsing (McDonald et al. , 2005).
---------------------------------------------------
P06-2098:182	108:155	In particuler, Single-best MIRA (McDonald et al, 2005) uses only the single margin constraint for the runner up y with the highest score.
---------------------------------------------------
P06-2098:183	107:155	(McDonald et al, 2005) presents this approach for dependency parsing.
---------------------------------------------------
H05-1066:184	214:223	To learn these structures we used online large-margin learning (McDonald et al. , 2005) that empirically provides state-of-the-art performance for Czech.
---------------------------------------------------
H05-1066:185	5:223	We evaluate these methods on the Prague Dependency Treebank using online large-margin learning techniques (Crammer et al. , 2003; McDonald et al. , 2005) and show that MST parsing increases efficiency and accuracy for languages with non-projective dependencies.
---------------------------------------------------
H05-1066:186	104:223	This algorithm has a runtime of O(n3) and has been employed successfully in both generative and discriminative parsing models (Eisner, 1996; McDonald et al. , 2005).
---------------------------------------------------
H05-1066:187	181:223	McD2005: The projective parser of McDonald et al.
---------------------------------------------------
D07-1033:188	73:389	Such methods include ALMA (Gentile, 2001) used in (Daume III and Marcu, 2005)2, MIRA (Crammer et al. , 2006) used in (McDonald et al. , 2005), and Max-Margin Markov Networks (Taskar et al. , 2003).
---------------------------------------------------
D07-1033:189	230:389	7 Discussion As we mentioned, there are some algorithms similar to ours (Collins and Roark, 2004; Daume III and Marcu, 2005; McDonald and Pereira, 2006; Liang et al. , 2006).
---------------------------------------------------
E06-1038:190	110:236	To do this we parse every sentence twice, once with a dependency parser (McDonald et al. , 2005b) and once with a phrase-structure parser (Charniak, 2000).
---------------------------------------------------
E06-1038:191	231:236	A possible direction of research is to investigate multilabel learning techniques for structured data (McDonald et al. , 2005a) that learn a scoring function separating a set of valid answers from all invalid answers.
---------------------------------------------------
E06-1038:192	70:236	This algorithm is really an extension of Viterbi to the case when scores factor over dynamic substrings of the text (Sarawagi and Cohen, 2004; McDonald et al. , 2005a).
---------------------------------------------------
D09-1086:193	106:269	We trained an edge-factored dependency parser (McDonald et al., 2005) on source domain data that followed one set of dependency conventions.
---------------------------------------------------
D09-1086:194	80:269	In the conditional models of 3 and 6, these features are those of an edge-factored dependency parser (McDonald et al., 2005).
---------------------------------------------------
P09-1053:195	171:248	8http://svmlight.joachims.org 9Our replication of the Wan et al. model is approximate, because we used different preprocessing tools: MXPOST for POS tagging (Ratnaparkhi, 1996), MSTParser for parsing (McDonald et al., 2005), and Dan Bikels interface (http://www.cis.upenn.edu/dbikel/ software.html#wn) to WordNet (Miller, 1995) for lemmatization information.
---------------------------------------------------
W06-2934:196	22:135	2.1 Decoding Instead of using the MST algorithm (McDonald et al. , 2005b) to maximise equation 1, we present an equivalent ILP formulation of the problem.
---------------------------------------------------
W06-2934:197	2:135	c2006 Association for Computational Linguistics Multi-lingual Dependency Parsing with Incremental Integer Linear Programming Sebastian Riedel and Ruket C akc and Ivan Meza-Ruiz ICCS School of Informatics University of Edinburgh Edinburgh, EH8 9LW, UK S.R.Riedel,R.Cakici,I.V.Meza-Ruiz@sms.ed.ac.uk Abstract Our approach to dependency parsing is based on the linear model of McDonald et al.(McDonald et al. , 2005b).
---------------------------------------------------
W06-2934:198	7:135	Our parser is inspired by McDonald et al.(2005a) which treats the task as the search for the highest scoring Maximum Spanning Tree (MST) in a graph.
---------------------------------------------------
N09-1068:199	179:220	We used the same features as (McDonald et al., 2005), augmented with information about whether or not a dependent is the first dependent (information they did not have).
---------------------------------------------------
D07-1014:200	9:189	Dependency parsing can be used to provide a bare bones syntactic structurethatapproximatessemantics,andithastheadditionaladvantageofadmittingfastparsingalgorithms (Eisner, 1996; McDonald et al. , 2005b) with a negligible grammar constant in many cases.
---------------------------------------------------
D07-1014:201	59:189	133 than the margin (McDonald et al. , 2005a).
---------------------------------------------------
D07-1014:202	10:189	The latest state-of-the-art statistical dependency parsers are discriminative, meaning that they are based on classifiers trained to score trees, given a sentence, either via factored whole-structure scores (McDonald et al. , 2005a) or local parsing decision scores (Hall et al. , 2006).
---------------------------------------------------
D09-1060:203	116:253	To match previous work (McDonald et al., 2005; McDonald and Pereira, 2006; Koo et al., 2008), we split the data into a training set (sections 2-21), a development set (Section 22), and a test set (section 23).
---------------------------------------------------
W09-1412:204	52:85	We further used some simple features from syntactic phrases (OpenNLP4 parser) and dependency parse trees (McDonald et al., 2005), extracted using parsers trained on Genia corpora.
---------------------------------------------------
W09-1412:205	22:85	We used a one-best version of MIRA (Crammer, 2004; McDonald et al., 2005) to choose w. MIRA is an online learning algorithm that updates the weight vector w for each training sentence xi according to the following rule: 95 wnew = argmin w bardblw  woldbardbl s.t. w  f(xi,yi)  w  f(x, y)  L(yi, y) where L(yi,y) is a measure of the loss of using y instead of the correct labeling yi, and y is a shorthand for ywold(xi).
---------------------------------------------------
P05-1067:206	199:217	The recent advances in parsing have achieved parsers with 3 ()On time complexity without the grammar constant (McDonald et al. , 2005).
---------------------------------------------------
W06-2937:207	20:149	2 System Description 241 Over the past decades, many state-of-the-art parsing algorithm were proposed, such as head-word lexicalized PCFG (Collins, 1998), Maximum Entropy (Charniak, 2000), Maximum/Minimum spanning tree (MST) (McDonald et al. , 2005), Bottom-up deterministic parsing (Yamada and Matsumoto, 2003), and Constant-time deterministic parsing (Nivre, 2003).
---------------------------------------------------
P08-1068:208	103:218	(1999), we used a coarsened version of the Czech part of speech tags; this choice also matches the conditions of previous work (McDonald et al., 2005b; McDonald and Pereira, 2006).
---------------------------------------------------
P08-1068:209	94:218	The data partition and head rules were chosen to match previous work (Yamada and Matsumoto, 2003; McDonald et al., 2005a; McDonald and Pereira, 2006).
---------------------------------------------------
P08-1068:210	152:218	Czech dependency structures may contain nonprojective edges, so we employ a maximum directed spanning tree algorithm (Chu and Liu, 1965; Edmonds, 1967; McDonald et al., 2005b) as our firstorder parser for Czech.
---------------------------------------------------
P08-1068:211	38:218	For many different part factorizations and structure domains Y(), it is possible to solve the above maximization efficiently, and several recent efforts have concentrated on designing new maximization algorithms with increased contextsensitivity (Eisner, 2000; McDonald et al., 2005b; McDonald and Pereira, 2006; Carreras, 2007).
---------------------------------------------------
P08-1068:212	54:218	The feature sets we used are similar to other feature sets in the literature (McDonald et al., 2005a; Carreras, 2007), so we will not attempt to give a exhaustive description of the features in this section.
---------------------------------------------------
P08-1068:213	114:218	In addition, in English parsing we ignore the parent-predictions of punctuation tokens,13 and in Czech parsing we retain the punctuation tokens; this matches previous work (Yamada and Matsumoto, 2003; McDonald et al., 2005a; McDonald and Pereira, 2006).
---------------------------------------------------
P08-1068:214	96:218	To facilitate comparisons with previous work (McDonald et al., 2005b; McDonald and Pereira, 2006), we used the training/development/test partition defined in the corpus and we also used the automatically-assigned part of speech tags provided in the corpus.10 Czech word clusters were derived from the raw text section of the PDT 1.0, which contains about 39 million words of newswire text.11 We trained the parsers using the averaged perceptron (Freund and Schapire, 1999; Collins, 2002), which represents a balance between strong performance and fast training times.
---------------------------------------------------
W09-1212:215	29:142	These features are borrowed from existing and widely-known systems (Xue and Palmer, 2004; McDonald et al., 2005; Carreras et al., 2006; Surdeanu et al., 2007).
---------------------------------------------------
D09-1127:216	148:211	Table 4 compares our baseline against the state-of-the-art graph-based (McDonald et al., 2005) and transition-based (Zhang and Clark, 2008) approaches, and confirms that our system performs at the same level with those stateof-the-art, and runs extremely fast in the deterministic mode (k=1), and still quite fast in the beamsearch mode (k=16).
---------------------------------------------------
P08-1007:217	129:175	In our work, we train the MSTParser4 (McDonald et al., 2005) on the Penn Treebank Wall Street Journal (WSJ) corpus, and use it to extract dependency relations from a sentence.
---------------------------------------------------
D07-1015:218	11:443	These structures are equivalent to non-projective dependency parses (McDonald et al. , 2005b), and more generally could be relevant to any task that involves learning a mapping from a graph to an underlying spanning tree.
---------------------------------------------------
D07-1015:219	22:443	This contrasts with the online learning algorithms used in previous work with spanning-tree models (McDonald et al. , 2005b).
---------------------------------------------------
D07-1127:220	9:167	One promising approach is based on exact search and structural learning (McDonald et al. , 2005; McDonald and Pereira, 2006).
---------------------------------------------------
