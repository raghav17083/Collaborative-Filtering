W05-0615:1	20:205	Klein and Manning (2001; 2002) recently achieved more encouraging results using an EM-like algorithm to induce syntactic constituent grammars, based on a deficient probability model.
---------------------------------------------------
P09-1009:2	155:226	Following (Klein and Manning, 2002), we restrict our model to binary trees, though we note that the alignment trees do not follow this restriction.
---------------------------------------------------
P09-1009:3	15:226	This task has been extensively studied in a monolingual setting and has proven to be difficult (Charniak and Carroll, 1992; Klein and Manning, 2002).
---------------------------------------------------
P09-1009:4	39:226	While PCFGs perform poorly on this task, the CCM model (Klein and Manning, 2002) has achieved large gains in performance and is among the state-of-the-art probabilistic models for unsupervised constituency parsing.
---------------------------------------------------
P09-1009:5	198:226	To evaluate both our model as well as the baseline, we use (unlabeled) bracket precision, recall, and F-measure (Klein and Manning, 2002).
---------------------------------------------------
P09-1009:6	36:226	In all cases, our model outperforms a state-of-the-art baseline: the Constituent Context Model (CCM) (Klein and Manning, 2002), sometimes by substantial margins.
---------------------------------------------------
P09-1009:7	179:226	In both cases our implementation achieves F-measure in the range of 69-70% on WSJ10, broadly in line with the performance reported by Klein and Manning (2002).
---------------------------------------------------
P09-1009:8	73:226	3.2 Model overview As our basic model of syntactic structure, we adopt the Constituent-Context Model (CCM) of Klein and Manning (2002).
---------------------------------------------------
P09-1009:9	38:226	2 Related Work The unsupervised grammar induction task has been studied extensively, mostly in a monolingual setting (Charniak and Carroll, 1992; Stolcke and Omohundro, 1994; Klein and Manning, 2002; Seginer, 2007).
---------------------------------------------------
P09-1009:10	169:226	During preprocessing of the corpora we remove all punctuation marks and special symbols, following the setup in previous grammar induction work (Klein and Manning, 2002).
---------------------------------------------------
P07-1049:11	9:181	The past few years have seen considerable improvement in the performance of unsupervised parsers (Klein and Manning, 2002; Klein and Manning, 2004; Bod, 2006a; Bod, 2006b) and, for the first time, unsupervised parsers have been able to improve on the right-branching heuristic for parsing English.
---------------------------------------------------
P07-1049:12	156:181	When Klein and Manning induce the parts-of-speech, they do so from a much larger corpus containing the full WSJ treebank together with additional WSJ newswire (Klein and Manning, 2002).
---------------------------------------------------
W06-2917:13	189:201	A number of statistical models have been proposed over the last few years by researchers such as (Klein and Manning, 2002; Klein and Manning, 2004) and (Solan et al. , 2005).
---------------------------------------------------
C08-1091:14	58:220	These include the constituentcontext model (CCM) (Klein and Manning, 2002), its extension using a dependency model (Klein and Manning, 2004), (U)DOP based models (Bod, 2006a; Bod, 2006b; Bod, 2007), an exemplar based approach (Dennis, 2005), guiding EM using contrastive estimation (Smith and Eisner, 2006), and the incremental parser of (Seginer, 2007).
---------------------------------------------------
P04-1062:15	196:264	An excellent recent result is by Klein and Manning (2002).
---------------------------------------------------
P04-1062:16	214:264	All conditions but 11We refer readers to Klein and Manning (2002) or Cover and Thomas (1991, p. 72) for details; computing expected counts for a sentence is a closed form operation.
---------------------------------------------------
P04-1062:17	220:264	The third line corresponds to the setup reported by Klein and Manning (2002).
---------------------------------------------------
C08-1128:18	69:207	Similar deficient models have been used very successfully before, for example, in the IBM models 36 and in the unsupervised grammar induction model of (Klein and Manning, 2002).
---------------------------------------------------
E09-1080:19	24:186	Our work fits well with several recent approaches aimed at completely unsupervised learning of the key aspects of syntactic structure: lexical categories (Schutze, 1993), phrase-structure (Klein and Manning, 2002; Seginer, 2007), phrasal categories (Borensztajn and Zuidema, 2007; Reichart and Rappoport, 2008) and dependencies (Klein and Manning, 2004).
---------------------------------------------------
P04-1061:20	135:208	Clark (2001) and Klein and Manning (2002) show that this approach can be successfully used for discovering syntactic constituents as well.
---------------------------------------------------
P04-1061:21	151:208	Klein and Manning (2002) gives comparative numbers showing that the basic CCM outperforms other recent systems on the ATIS corpus (which many other constituency induction systems have reported on).
---------------------------------------------------
P04-1061:22	194:208	is worth noting that these totally unsupervised numbers are better than the performance of the CCM model of Klein and Manning (2002) running off of Penn treebank word classes.
---------------------------------------------------
P04-1061:23	7:208	An important distinction should be drawn between work primarily interested in the weak generative capacity of models, where modeling hierarchical structure is only useful insofar as it leads to improved models over observed structures (Baker, 1979; Chen, 1995), and work interested in the strong generative capacity of models, where the unobserved structure itself is evaluated (van Zaanen, 2000; Clark, 2001; Klein and Manning, 2002).
---------------------------------------------------
P04-1061:24	125:208	As we will see, this combined model finds correct dependencies more successfully than the model above, and finds constituents more successfully than the model of Klein and Manning (2002).
---------------------------------------------------
P04-1061:25	139:208	In Klein and Manning (2002), we proposed a constituent-context model (CCM) which solves this problem by building constituency decisions directly into the distributional model, by earmarking a single cluster d for non-constituents.
---------------------------------------------------
P04-1061:26	124:208	To this end, after briefly recapping the model of Klein and Manning (2002), we present a combined model that exploits dependencies and constituencies.
---------------------------------------------------
P04-1061:27	158:208	8In Klein and Manning (2002), we reported results using unlabeled bracketing statistics which gave no credit for brackets which spanned the entire sentence (raising the scores) but macro-averaged over sentences (lowering the scores).
---------------------------------------------------
P04-1061:28	152:208	While absolute numbers are hard to compare across corpora, all the systems compared to in Klein and Manning (2002) parsed below a right-branching baseline, while the CCM is substantially above it.
---------------------------------------------------
P04-1061:29	10:208	2 Unsupervised Dependency Parsing Most recent progress in unsupervised parsing has come from tree or phrase-structure grammar based models (Clark, 2001; Klein and Manning, 2002), but there are compelling reasons to reconsider unsupervised dependency parsing.
---------------------------------------------------
W09-1108:30	65:257	These include CCM (Klein and Manning, 2002), the DMV and DMV+CCM models (Klein and Manning, 2004), (U)DOP based models (Bod, 2006a; Bod, 2006b; Bod, 2007), an exemplar based approach (Dennis, 2005), guiding EM using contrastive estimation (Smith and Eisner, 2006), and the incremental parser of Seginer (2007) which we use here.
---------------------------------------------------
P07-3008:31	90:172	Evaluation of the algorithm is done according to PARSEVAL, except for a few changes that are also proposed by Klein and Manning (2002).
---------------------------------------------------
P07-3008:32	109:172	Still, Klein and Manning (2002) and Bod (2006) stick to tag-based models.
---------------------------------------------------
P07-3008:33	21:172	As an attempt to ameliorate this, and as an attempt to confirm Klein and Mannings (2002) and Bods (2006) thesis that good enough unsupervised POS-taggers exist to justify using POS-tags instead of words in evaluating GI systems, I pre43 sented the algorithms with both POS-tags that were induced from Biemanns unsupervised POS-tagging algorithm and hand-corrected POS-tags.
---------------------------------------------------
N09-1012:34	178:217	Following Klein and Manning (2002), sentences longer than 10 words after removing punctuation are ignored.
---------------------------------------------------
W06-2912:35	103:174	To make our parse results comparable to those of Klein and Manning (2002, 2004, 2005), we will use exactly the same evaluation metrics for unlabeled precision (UP) and unlabeled recall (UR), defined in Klein (2005: 21-22).
---------------------------------------------------
W06-2912:36	12:174	Clark (2001) reports 42.0% unlabeled f-score on the same data using distributional clustering, and Klein and Manning (2002) obtain 51.2% unlabeled f-score on ATIS part-of-speech strings using a constituent-context model called CCM.
---------------------------------------------------
W06-2912:37	110:174	We next tested UDOP on two additional domains from Chinese and German which were also used in Klein and Manning (2002, 2004): the Chinese treebank (Xue et al. 2002) and the NEGRA corpus (Skut et al. 1997).
---------------------------------------------------
W06-2912:38	13:174	Moreover, on Penn Wall Street Journal p-os-strings  10 (WSJ10), Klein and Manning (2002) report 71.1% unlabeled f-score.
---------------------------------------------------
W06-2912:39	113:174	Table 1 shows the results of U-DOP in terms of UP, UR and F1 compared to the results of the CCM model by Klein and Manning (2002), the DMV dependency learning model by Klein and Manning (2004) together with their combined model DMV+CCM.
---------------------------------------------------
W06-2912:40	43:174	As shown by Klein and Manning (2002, 2004), the extension to inducing trees for words instead of p-o-s tags is rather straightforward since there exist several unsupervised part-of-speech taggers with high accuracy, which can be combined with unsupervised parsing (see e.g. Schtze 1996; Clark 2000).
---------------------------------------------------
P05-1025:41	57:182	Although we need manually labeled data to train the classifier for labeling dependencies, the size of this training set is far smaller than what would be necessary to train a parser to find labeled dependen3Klein and Manning (2002) offer an informal argument that constituent labels are much more easily separable in multidimensional space than constituents/distituents.
---------------------------------------------------
P06-1111:42	31:194	As in previous work, we begin with the part-of-speech (POS) tag sequences for each sentence rather than lexical sequences (Carroll and Charniak, 1992; Klein and Manning, 2002).
---------------------------------------------------
P06-1111:43	6:194	Toimprovethequality of the induced trees, we combine our PCFG induction with the CCM model of Klein and Manning (2002), which has complementary stengths: it identifies brackets but does not label them.
---------------------------------------------------
P06-1111:44	87:194	This idea has been repeatedly and successfully operationalized using various kinds of distributional clustering, where we define a similarity measure between two items on the basis of their immediate left and right contexts (Schutze, 1995; Clark, 2000; Klein and Manning, 2002).
---------------------------------------------------
P06-1111:45	53:194	Klein and Manning (2002) suggest that the task of labeling constituents is significantly easier than identifying them.
---------------------------------------------------
P06-1111:46	24:194	Finally, weintersectthefeature-augmentedPCFGwiththe CCM model of Klein and Manning (2002), a highquality bracketing model.
---------------------------------------------------
P06-1111:47	9:194	Recent work has successfully induced unlabeled grammatical structure, but has not successfully learned labeled tree structure (Klein and Manning, 2002; Klein and Manning, 2004; Smith and Eisner, 2004).
---------------------------------------------------
P06-1111:48	161:194	Another type of error also reported by Klein and Manning (2002) is MD VB groupings in infinitival VPs also sometimes argued by linguists (Halliday, 2004).
---------------------------------------------------
P06-1111:49	190:194	They also achieve the best reported unlabeled F1 measure.8 Another positive property of this approach is that it tries to reconcile the success of distributional clustering approaches to grammar induction (Clark, 2001; Klein and Manning, 2002), with the CFG tree models in the supervised literature (Collins, 1999).
---------------------------------------------------
P06-1111:50	117:194	One such model is the constituent-context model (CCM) of Klein and Manning (2002), a generative distributional model.
---------------------------------------------------
P04-1060:51	157:158	A number of studies are related to the work we presented, most specifically work on parallel-text based information projection for parsing (Hwa et al. , 2002), but also grammar induction work based on constituent/distituent information (Klein and Manning, 2002) and (language-internal) alignmentbased learning (van Zaanen, 2000).
---------------------------------------------------
P04-1060:52	20:158	(Klein and Manning, 2002)).
---------------------------------------------------
P08-1061:53	23:148	Consequently, most previous work that has attempted semi-supervised or unsupervised approaches to parsing have not produced results beyond the state of the art supervised results (Klein and Manning, 2002; Klein and Manning, 2004).
---------------------------------------------------
W07-0207:54	68:214	Figure 3: A Nonprojective Dependency Graph 1 2 3 4 5 6 7 8 9 100 0.5 1 1.5 2 2.5 3 x 10 4 Words Distant Number of Dependencies Figure 4: Distance Between Dependents in WSJ10 4.3 Context The core of several UGI approaches is distributional analysis (Brill and Marcus, 1992; van Zaanen, 2000; Klein and Manning, 2002; Paskin, 2001; Klein and Manning, 2004; Solan et al. , 2005).
---------------------------------------------------
W07-0207:55	19:214	2 Latent semantics Previous work has focused on syntax to the exclusion of semantics (Brill and Marcus, 1992; van Zaanen, 2000; Klein and Manning, 2002; Paskin, 2001; Klein and Manning, 2004; Solan et al. , 2005).
---------------------------------------------------
W07-0207:56	12:214	Recent work (Klein and Manning, 2002; Klein and Manning, 2004) has renewed interest by using a UGI model to parse sentences from the Wall Street Journal section of the Penn Treebank (WSJ).
---------------------------------------------------
W09-1120:57	154:237	The idea of representing a constituent by its yield 160 and (a different definition of) context is used by the CCM unsupervised parsing model (Klein and Manning, 2002).
---------------------------------------------------
W09-1120:58	11:237	The last decade has seen significant progress in this field of research (Klein and Manning, 2002; Klein and Manning, 2004; Bod, 2006a; Bod, 2006b; Smith and Eisner, 2006; Seginer, 2007).
---------------------------------------------------
W09-1120:59	127:237	These include CCM (Klein and Manning, 2002), the DMV and DMV+CCM models (Klein and Manning, 2004), (U)DOP based mod159 0 50 1000 5000 10000 15000 t # of constituents appearing at least t times  0 50 1000 1000 2000 3000 4000 5000 6000 7000 8000 t   P = 5 P = 50 P = 5 P = 50 Figure 2: Number of constituents appearing at least t times (nc(t)) as a function of t. Shown are WSJ (left) and NEGRA (right), where constituents are represented according to PUPAs PCR with 5 POS tags (P = 5, solid line) or 50 POS tags (P = 50, dashed line).
---------------------------------------------------
C08-1042:60	145:202	While the tags are induced from all sentences in the section, following the practice in (Klein and Manning, 2002; Klein and Manning, 2004), we remove punctuation, and consider only sentences of length not greater than 10 in our grammar induction experiments.
---------------------------------------------------
C08-1042:61	37:202	We chose the baseline system primarily to match previous evaluations of grammar induction using induced tags (Klein and Manning, 2002).
---------------------------------------------------
C08-1042:62	33:202	We follow Klein and Manning (2002) in using Kmeans to cluster the d dimensional word vectors into parts-of-speech.
---------------------------------------------------
C08-1042:63	97:202	The Grammar Induction systems we use to evaluate the above taggers are the Constituent-Context Model (CCM), the Dependency Model with Valence (DMV), and a model which combines the two (CCM+DMV) outlined in (Klein and Manning, 2002; Klein and Manning, 2004).
---------------------------------------------------
C08-1042:64	98:202	3.1 Constituent Grammar Induction Klein and Manning (2002) present a generative model for inducing constituent boundaries from part-of-speech tagged text.
---------------------------------------------------
C08-1042:65	29:202	We implement the baseline system, which Klein and Manning (2002) use for their grammar induction experiments with induced part-of-speech tags.
---------------------------------------------------
C08-1042:66	103:202	We evaluate induced constituency trees against those of the Penn Treebank using the versions of unlabeled precision, recall, and F-score used by Klein and Manning (2002).
---------------------------------------------------
P06-1109:67	10:166	Where van Zaanen (2000) achieved 39.2% unlabeled f-score on ATIS word strings, Clark (2001) reports 42.0% on the same data, and Klein and Manning (2002) obtain 51.2% f-score on ATIS part-of-speech strings using a constituent-context model called CCM.
---------------------------------------------------
P06-1109:68	119:166	We next tested UML-DOP on two additional domains which were also used in Klein and Manning (2004) and Bod (2006): the German NEGRA10 (Skut et al. 1997) and the Chinese CTB10 (Xue et al. 2002) both containing 2200+ sentences  10 words after removing punctuation.
---------------------------------------------------
P06-1109:69	120:166	Table 1 shows the results of UML-DOP compared to U-DOP, the CCM model by Klein and Manning (2002), the DMV dependency learning model by Klein and Manning (2004) as well as their combined model DMV+CCM.
---------------------------------------------------
P06-1109:70	11:166	On Penn Wall Street Journal po-s-strings  10 (WSJ10), Klein and Manning (2002) report 71.1% unlabeled f-score with CCM.
---------------------------------------------------
P06-1109:71	49:166	Previous models like Klein and Manning's (2002, 2005) CCM model limit the dependencies to "contiguous subsequences of a sentence".
---------------------------------------------------
E09-1023:72	33:204	This shortest context heuristic receives some support from research on first language acquisition (Mintz, 2006) and unsupervised grammar induction (Klein and Manning, 2002).
---------------------------------------------------
P06-2022:73	161:187	It should be possible in the next few years to create a dependency parserforalanguagewithnoexistinglinguisticresources (Klein and Manning, 2002).
---------------------------------------------------
N06-1020:74	11:208	Finally, there are unsupervised strategies where no data is labeled and all annotations (including the grammar itself) must be discovered (Klein and Manning, 2002).
---------------------------------------------------
D09-1086:75	201:269	Much previous work on unsupervised grammar induction has used gold-standard partof-speech tags (Smith and Eisner, 2006b; Klein and Manning, 2004; Klein and Manning, 2002).
---------------------------------------------------
D07-1014:76	131:189	This has been applied widely in unsupervised parsing (Carroll and Charniak, 1992; Klein and Manning, 2002).
---------------------------------------------------
H05-1036:77	317:369	These techniques included unweighted FS morphology, conditional random fields (Lafferty et al. , 2001), synchronous parsers (Wu, 1997; Melamed, 2003), lexicalized parsers (Eisner and Satta, 1999),22 partially supervised training `a la (Pereira and Schabes, 1992),23 and grammar induction (Klein and Manning, 2002).
---------------------------------------------------
P07-1051:78	88:173	Table 1 shows the f-scores for U-DOP* and UML-DOP against the f-scores for U-DOP reported in Bod (2006), the CCM model in Klein and Manning (2002), the DMV dependency model in Klein and Manning (2004) and their combined model DMV+CCM.
---------------------------------------------------
P07-1051:79	18:173	Bod (2006) reports 82.9% unlabeled f-score on the same WSJ10 as used by Klein and Manning (2002, 2004).
---------------------------------------------------
P07-1051:80	78:173	4 Evaluation on hand-annotated corpora To evaluate U-DOP* against UML-DOP and other unsupervised parsing models, we started out with three corpora that are also used in Klein and Manning (2002, 2004) and Bod (2006): Penns WSJ10 which contains 7422 sentences  10 words after removing empty elements and punctuation, the German NEGRA10 corpus and the Chinese Treebank CTB10 both containing 2200+ sentences  10 words after removing punctuation.
---------------------------------------------------
P07-1051:81	12:173	Where van Zaanen (2000) and Clark (2001) induced unlabeled phrase structure for small domains like the ATIS, obtaining around 40% unlabeled f-score, Klein and Manning (2002) report 71.1% f-score on Penn WSJ part-of-speech strings  10 words (WSJ10) using a constituentcontext model called CCM.
---------------------------------------------------
P07-1051:82	82:173	We used the same evaluation metrics for unlabeled precision (UP) and unlabeled recall (UR) as in Klein and Manning (2002, 2004).
---------------------------------------------------
