N07-3002:1	7:74	1 Introduction Over the past decade, there has been tremendous progress on learning parsing models from treebank data (Magerman, 1995; Collins, 1999; Charniak, 1997; Ratnaparkhi, 1999; Charniak, 2000; Wang et al. , 2005; McDonald et al. , 2005).
---------------------------------------------------
N07-3002:2	10:74	Subsequent research began to focus more on conditional models of parse structure given the input sentence, which allowed discriminative training techniques such as maximum conditional likelihood (i.e. maximum entropy ) to be applied (Ratnaparkhi, 1999; Charniak, 2000).
---------------------------------------------------
D09-1119:3	37:258	While it was initially believed that lexicalization of PCFG parsers (Collins, 1997; Charniak, 2000) is crucial for obtaining good parsing results, Gildea (2001) demonstrated that the lexicalized Model-1 parser of Collins (1997) does not benefit from bilexical information when tested on a new text domain, and only marginally benefits from such information when tested on the same text domain as the training corpora.
---------------------------------------------------
P06-1051:4	160:203	We also used the following resources: The Charniak parser (Charniak, 2000) and the morphalemmatiser (Minnen et al. , 2001) to carry out the syntactic and morphological analysis.
---------------------------------------------------
W04-2003:5	45:197	Statistical disambiguation such as (Collins and Brooks, 1995) for PP-attachment or (Collins, 1997; Charniak, 2000) for generative parsing greatly improve disambiguation, but as they model by imitation instead of by understanding, complete soundness has to remain elusive.
---------------------------------------------------
W04-2003:6	106:197	Much of the interesting work is determining what goes into [the history] H(c)(Charniak, 2000).
---------------------------------------------------
W04-2003:7	28:197	A number of robust statistical parsers that oer solutions to these problems have now become available (Charniak, 2000; Collins, 1999; Henderson, 2003), but they typically produce CFG constituency data as output, trees that do not express long-distance dependencies.
---------------------------------------------------
W06-1601:8	132:193	5 Datasets and Evaluation We train our models with verb instances extracted from three parsed corpora: (1) the Wall Street Journal section of the Penn Treebank (PTB), which was parsed by human annotators (Marcus et al. , 1993), (2) the Brown Laboratory for Linguistic Information Processing corpus of Wall Street Journal text (BLLIP), which was parsed automatically by the Charniak parser (Charniak, 2000), and (3) the Gigaword corpus of raw newswire text (GW), which we parsed ourselves with the Stanford parser.
---------------------------------------------------
E09-1090:9	197:207	(2008) 87.8 88.2 88.0 >250* Petrov & Klein (2008) 88.3 3* Sagae & Lavie (2006) 87.8 88.1 87.9 17 Charniak & Johnson (2005) 90.6 91.3 91.0 Unk Tsuruoka & Tsujii (2005) 85.0 86.8 85.9 2 Collins (1999) 88.1 88.3 88.2 39** Tjong Kim Sang (2001) 78.7 82.3 80.5 Unk Charniak (2000) 89.6 89.5 89.5 23** Ratnaparkhi (1997) 86.3 87.5 86.9 Unk Table 6: Parsing performance on section 23 (all sentences).
---------------------------------------------------
E09-1090:10	59:207	However, there is still a large gap between the accuracy of chunking-based parsers and that of widely-used practical parsers such as Collins parser and Charniak parser (Collins, 1999; Charniak, 2000).
---------------------------------------------------
E09-1090:11	10:207	Generative models based on lexicalized PCFGs enjoyed great success as the machine learning framework for full parsing (Collins, 1999; Charniak, 2000), but recently discriminative models attract more attention due to their superior accuracy (Charniak and Johnson, 2005; Huang, 2008) and adaptability to new grammars and languages (Buchholz and Marsi, 2006).
---------------------------------------------------
P03-1013:12	222:266	TnT tagging Perfect tagging LR LP LR LP PP 3.45 1.60 4.21 3.35 S 1.28 0.11 2.23 1.22 Coord 1.87 0.39 1.54 0.80 VP 0.72 0.18 0.58 0.30 AP 0.57 0.10 0.08 0.07 AV P 0.32 0.44 0.10 0.11 NP 0.06 0.78 0.15 0.02 Table 6: Change in performance when reverting to head-head statistics for individual categories ter information (Charniak, 2000), as illustrated in Table 4.
---------------------------------------------------
P03-1013:13	171:266	sister head tag X Table 4: Linguistic features in the current model compared to the models of Carroll and Rooth (1998), Collins (1997), and Charniak (2000) Negra, based on Collinss (1997) model for nonrecursive NPs in the Penn Treebank (which are also flat).
---------------------------------------------------
P03-1013:14	177:266	Table 4 shows the linguistic features of the resulting model compared to the models of Carroll and Rooth (1998), Collins (1997), and Charniak (2000).
---------------------------------------------------
P03-1013:15	12:266	Lexicalization can increase parsing performance dramatically for English (Carroll and Rooth, 1998; Charniak, 1997, 2000; Collins, 1997), and the lexicalized model proposed by Collins (1997) has been successfully applied to Czech (Collins et al. , 1999) and Chinese (Bikel and Chiang, 2000).
---------------------------------------------------
P03-1013:16	55:266	3 Probabilistic Parsing Models 3.1 Probabilistic Context-Free Grammars Lexicalization has been shown to improve parsing performance for the Penn Treebank (e.g. , Carroll and Rooth 1998; Charniak 1997, 2000; Collins 1997).
---------------------------------------------------
P03-1013:17	7:266	1 Introduction Treebank-based probabilistic parsing has been the subject of intensive research over the past few years, resulting in parsing models that achieve both broad coverage and high parsing accuracy (e.g. , Collins 1997; Charniak 2000).
---------------------------------------------------
P03-1013:18	180:266	Charniaks (2000) model extends this to higher order Markov chains (first to third order), and therefore includes category information about previous sisters.The current model differs from all these proposals: it does not use any information about the head sister, but instead includes the category, head word, and head tag of the previous sister, effectively treating it as the head.
---------------------------------------------------
J07-4004:19	199:948	Long-range dependencies are relatively common in text such as newspaper text, but are typically not recovered by treebank parsers such as Collins (2003) and Charniak (2000).
---------------------------------------------------
W07-1001:20	121:186	4.3 Parsing For automatic parsing, we made use of the wellknownCharniakparser(Charniak,2000).
---------------------------------------------------
P02-1042:21	5:157	1 Introduction Most recent wide-coverage statistical parsers have used models based on lexical dependencies (e.g. Collins (1999), Charniak (2000)).
---------------------------------------------------
C02-1138:22	77:196	The other kind of treebank is the BLLIP corpus (Charniak, 2000).
---------------------------------------------------
N07-1053:23	85:200	parse-based features, we parse the TEXT elements of our documents with the Charniak parser (Charniak, 2000).
---------------------------------------------------
W06-3806:24	76:86	We also used the following resources: the Charniak parser (Charniak, 2000) to carry out the syntactic analysis; the wn::similaritypackage (Pedersen et al. , 2004) to compute the Jiang&Conrath (J&C) distance (Jiang and Conrath, 1997) needed to implement the lexical similarity siml(T,H) as defined in (Corley and Mihalcea, 2005); SVM-lightTK (Moschitti, 2004) to encode the basic tree kernel function, KT, in SVM-light (Joachims, 1999).
---------------------------------------------------
P06-2041:25	160:166	For the English data, the result for SVM with model 5 is about 3 percentage points below the results obtained with the parser of Charniak (2000) and reported by Yamada and Matsumoto (2003).
---------------------------------------------------
P06-2041:26	6:166	1 Introduction Mainstream approaches in statistical parsing are based on nondeterministic parsing techniques, usually employing some kind of dynamic programming, in combination with generative probabilistic models that provide an n-best ranking of the set of candidate analyses derived by the parser (Collins, 1997; Collins, 1999; Charniak, 2000).
---------------------------------------------------
P04-1040:27	184:199	TiMBL performed well on tasks where structured, more complicated and task-specific statistical models have been used previously (Blaheta and Charniak, 2000).
---------------------------------------------------
P04-1040:28	101:199	Thus, it is informative to compare our results with those reported in (Blaheta and Charniak, 2000) for this same task.
---------------------------------------------------
P04-1040:29	79:199	As an alternative to hardcoded heuristics, Blaheta and Charniak (2000) proposed to recover the Penn functional tags automatically.
---------------------------------------------------
P04-1040:30	15:199	The evaluation of the transformed output of the parsers of Charniak (2000) and Collins (1999) gives 90% unlabelled and 84% labelled accuracy with respect to dependencies, when measured against a dependency corpus derived from the Penn Treebank.
---------------------------------------------------
P04-1040:31	110:199	First, because of the different definition of a correctly identified constituent in the parsers output, we apply our method to a greater portion of all labels produced by the parser (95% vs. 89% reported in (Blaheta and Charniak, 2000)).
---------------------------------------------------
P04-1040:32	27:199	Blaheta and Charniak (2000) presented the first method for assigning Penn functional tags to constituents identified by a parser.
---------------------------------------------------
P04-1040:33	31:199	In the experiments we used the parsers described in (Charniak, 2000) and (Collins, 1999).
---------------------------------------------------
E09-1027:34	108:237	Petersen and Ostendorf (2009) included features calculated from parsing the sentences in their corpus using the Charniak parser (Charniak, 2000): average parse tree height, average number of noun phrases per sentence, average number of verb phrases per sentence, and average number of SBARs per sentence.
---------------------------------------------------
W06-3119:35	20:125	We then use Charniaks parser (Charniak, 2000) to generate the most likely parse tree for each English target sentence in the training corpus.
---------------------------------------------------
D07-1117:36	116:217	First a bracketing parser (the Charniak parser (Charniak, 2000) in our case) is used to generate the parse tree of a sentence, then the const2dep tool developed by Hwa was utilized to convert the bracketing tree to a dependency tree based on the head percolation table developed by the second author.
---------------------------------------------------
N06-1024:37	76:144	7And the (Charniak, 2000) parser that (Blaheta, 2003) used has a reported F-measure of 89.5, higher than the Bikel parser used here.
---------------------------------------------------
N06-1024:38	8:144	Modern statistical parsers such as (Collins, 2003) and (Charniak, 2000) however ignore much of this information and return only an We would like to thank Fernando Pereira, Dan Bikel, Tony Kroch and Mark Liberman for helpful suggestions.
---------------------------------------------------
N06-1024:39	67:144	For purposes of comparison, we have calculated our overall score both with and without CLR.5 The (Blaheta, 2003) numbers in parentheses in Table 1 are from hisfeaturetreesspecializedfortheSyntacticandSemantic groups, while all his other numbers, including the overall score, are from using a single feature set for his four function tag groups.6 5(Jijkoun and de Rijke, 2004) do not state whether they are including CLR, but since they are comparing their results to (Blaheta and Charniak, 2000), we are assuming that they do.
---------------------------------------------------
N06-1024:40	122:144	Note that our systems parsed scores were obtained using our modified version of Bikels implementation of Collinss thesis parser which assigns function tags, while the other PSLB postprocessing systems use Charniaks parser (Charniak, 2000) and Dienes and Dubey integrate empty category recovery directly into a variant of Collinss parser.
---------------------------------------------------
P03-1055:41	7:180	However, such constructions prove to be difficult for stochastic parsers (Collins et al. , 1999) and they either avoid tackling the problem (Charniak, 2000; Bod, 2003) or only deal with a subset of the problematic cases (Collins, 1997).
---------------------------------------------------
P01-1010:42	50:160	4.1 The base line For our base line parse accuracy, we used the now standard division of the WSJ (see Collins 1997, 1999; Charniak 1997, 2000; Ratnaparkhi 1999) with sections 2 through 21 for training (approx.
---------------------------------------------------
P01-1010:43	67:160	The table shows that by using the base line subtree set, our parser outperforms most previous parsers but it performs worse than the parser in Charniak (2000).
---------------------------------------------------
P01-1010:44	158:160	The importance of including single nonheadwords is now also uncontroversial (e.g. Collins 1997, 1999; Charniak 2000), and the current paper has shown the importance of including two and more nonheadwords.
---------------------------------------------------
P01-1010:45	141:160	Charniak 1996, 1997), while most current stochastic parsing models use a "markov grammar" (e.g. Collins 1999; Charniak 2000).
---------------------------------------------------
P01-1010:46	80:160	The highest scores are obtained if the full base line subtree set is used, but they remain behind the results of Charniak (2000).
---------------------------------------------------
P01-1010:47	66:160	Table 1 shows the LP and LR scores obtained with our base line subtree set, and compares these scores with those of previous stochastic parsers tested on the WSJ (respectively Charniak 1997, Collins 1999, Ratnaparkhi 1999, and Charniak 2000).
---------------------------------------------------
P01-1010:48	87:160	Our highest scores of 90.8% LP and 90.5% LR outperform the scores of the best previously published parser by Charniak (2000) who obtains 90.1% for both LP and LR.
---------------------------------------------------
P01-1010:49	123:160	These scores slightly outperform the best previously published parser by Charniak (2000), who obtained 89.5% LP and 89.6% LR for test sentences  100 words.
---------------------------------------------------
P01-1010:50	69:160	While most subtree restrictions diminish the accuracy scores, we will see that there are restrictions that improve our scores, even beyond those of Charniak (2000).
---------------------------------------------------
P01-1010:51	105:160	Head-lexicalized stochastic grammars have recently become increasingly popular (see Collins 1997, 1999; Charniak 1997, 2000).
---------------------------------------------------
P01-1010:52	154:160	context-free rules Charniak (1996) Collins (1996), Eisner (1996) context-free rules, headwords Charniak (1997) context-free rules, headwords, grandparent nodes Collins (2000) context-free rules, headwords, grandparent nodes/rules, bigrams, two-level rules, two-level bigrams, nonheadwords Bod (1992) all fragments within parse trees Scope of Statistical Dependencies Model Figure 4.
---------------------------------------------------
P04-1082:53	62:164	Charniaks parser (Charniak, 2000), however, does not include function tags, so in order for the algorithm to work properly on parser output (see Section 5), additional functions were written to approximate the required tags.
---------------------------------------------------
P04-1082:54	93:164	5.2 Parser output The system was also run using as input the output of Charniaks parser (Charniak, 2000).
---------------------------------------------------
P04-1082:55	6:164	1 Introduction Many recent approaches to parsing (e.g. Charniak, 2000) have focused on labeled bracketing of the input string, ignoring aspects of structure that are not reflected in the string, such as phonetically null elements and long-distance dependencies, many of which provide important semantic information such as predicate-argument structure.
---------------------------------------------------
P04-1082:56	80:164	5 Evaluation Following Johnson (2002), the system was evaluated on two different kinds of input: first, on perfect input, i.e., PTB annotations stripped of all empty categories and information related to them; and second, on imperfect input, in this case the output of Charniaks (2000) parser.
---------------------------------------------------
P04-1082:57	12:164	State-of-the-art statistical parsers (e.g. Charniak, 2000) typically provide a labeled bracketing only; i.e., a parse tree without empty categories.
---------------------------------------------------
P04-1082:58	63:164	Presumably, the accuracy of the algorithm on parser output would be enhanced by accurate prior assignment of the tags to all relevant nodes, as in Blaheta and Charniak (2000) (see also Section 5).
---------------------------------------------------
P04-1082:59	101:164	As mentioned in Section 4, it is believed that the results of the current method on parser output would improve if that output were reliably assigned function tags, perhaps along the lines of Blaheta and Charniak (2000).
---------------------------------------------------
N01-1029:60	117:249	The second set was trained on the BLLIP WSJ Corpus (BWC), which is a machine-parsed (Charniak, 2000) version of (a selection of) the ACL/DCI corpus, very similar to the selection made for the WSJ0/1 CSR corpus.
---------------------------------------------------
W05-1509:61	28:202	Following (Blaheta and Charniak, 2000), we concentrate on syntactic and semantic function labels.
---------------------------------------------------
W05-1509:62	155:202	Following (Blaheta and Charniak, 2000), incorrectly parsed constituents will be ignored (roughly 11% of the total) in the evaluation of the precision and recall of the function labels, but not in the evaluation of the parser.
---------------------------------------------------
W05-1509:63	153:202	Individual performance on syntactic and semantic function labelling compare favourably to previous attempts (Blaheta, 2004; Blaheta and Charniak, 2000).
---------------------------------------------------
W05-1509:64	11:202	duce trees annotated with bare phrase structure labels (Collins, 1999; Charniak, 2000).
---------------------------------------------------
W05-1509:65	165:202	Previous work that uses, like us, a single model for both types of labels reaches an F measure of 95.7% for syntactic labels and 79.0% for semantic labels (Blaheta and Charniak, 2000).
---------------------------------------------------
W05-1509:66	52:202	As with many other statistical parsers (Collins, 1999; Charniak, 2000), SSN parsers use a history-based model of parsing.
---------------------------------------------------
W05-1509:67	14:202	Unlike phrase structure labels, function labels are contextdependent and encode a shallow level of phrasal and lexical semantics, as observed first in (Blaheta and Charniak, 2000).
---------------------------------------------------
W05-1509:68	25:202	Specifically, the parser outputs additional labels indicating the function of a constituent in the tree, such as NP-SBJ or PP-TMP in the tree 1(Blaheta and Charniak, 2000) talk of function tags.We will instead use the term function label, to indicate function identifiers, as they can decorate any node in the tree.
---------------------------------------------------
W07-0605:69	19:174	(Charniak, 2000; Briscoe et al. , 2006), have wide coverage and yield grammatical representations capable of supporting various applications (e.g. summarization, information extraction).
---------------------------------------------------
P05-1072:70	21:208	We investigate ways to combine hypotheses generated from semantic role taggers trained using different syntactic views  one trained using the Charniak parser (Charniak, 2000), another on a rule-based dependency parser  Minipar (Lin, 1998), and a third based on a flat, shallow syntactic chunk representation (Hacioglu, 2004a).
---------------------------------------------------
W07-0604:71	130:187	(2005) report 86.9% LAS on about 2,000 words of Eve data, using the Charniak (2000) parser with a separate dependency-labeling step.
---------------------------------------------------
W07-0604:72	177:187	That work relied on a phrase-structure statistical parser (Charniak, 2000) trained on the Penn Treebank, and the output of that parser had to be converted into CHILDES grammatical relations.
---------------------------------------------------
W07-0604:73	90:187	(2004) has been used in practice for automatic parsing of child language transcripts (Sagae et al. , 2004; Sagae et al. , 2005), such work relied mainly on a statistical parser (Charniak, 2000) trained on the Wall Street Journal portion of the Penn Treebank, since a large enough corpus of annotated CHILDES data was not available to train a domain-specific parser.
---------------------------------------------------
P06-1041:74	6:175	Stochastic parsers for English trained on the Penn Treebank have peaked their performance around 90% (Charniak, 2000).
---------------------------------------------------
W04-0834:75	50:87	3.4 Syntactic Relations We first parse the sentence containinga0 with a statistical parser (Charniak, 2000).
---------------------------------------------------
W08-0909:76	36:236	Rather than using sentence lengthasa proxy, measurescanemploy toolsforautomatic analysis of the syntactic structure of texts (e.g., (Charniak, 2000)).
---------------------------------------------------
W05-1516:77	140:171	Performance of Alternative Models 157 5 Related Work Previous parsing models (e.g. , Collins, 1997; Charniak, 2000) maximize the joint probability P(S, T) of a sentence S and its parse tree T. We maximize the conditional probability P(T | S).
---------------------------------------------------
P09-2064:78	75:110	Charniak parser (Charniak, 2000) is used for POS tagging and full parsing.
---------------------------------------------------
P06-3009:79	49:147	Parsing models have been developed for different languages and state-of-the-art results have been reported for, e.g., English (Collins, 1997; Charniak, 2000).
---------------------------------------------------
W06-2904:80	8:160	1 Introduction Over the past decade, there has been tremendous progress on learning parsing models from treebank data (Collins, 1997; Charniak, 2000; Wang et al. , 2005; McDonald et al. , 2005).
---------------------------------------------------
W06-2904:81	11:160	Subsequent research began to focus more on conditional models of parse structure given the input sentence, which allowed discriminative training techniques such as maximum conditional likelihood (i.e. maximum entropy ) to be applied (Ratnaparkhi, 1999; Charniak, 2000).
---------------------------------------------------
N09-1062:82	8:209	Considerable effort is required to coax good results from a PCFG, in the form of grammar engineering, feature selection and clever smoothing (Collins, 1999; Charniak, 2000; Charniak and Johnson, 2005; Johnson, 1998).
---------------------------------------------------
E09-1048:83	132:218	Proper nouns and other parts of speech were identified running Charniaks parser (Charniak, 2000) on the news articles.
---------------------------------------------------
P08-1006:84	7:181	1 Introduction Parsing technologies have improved considerably in the past few years, and high-performance syntactic parsers are no longer limited to PCFG-based frameworks (Charniak, 2000; Klein and Manning, 2003; Charniak and Johnson, 2005; Petrov and Klein, 2007), but also include dependency parsers (McDonald and Pereira, 2006; Nivre and Nilsson, 2005; Sagae and Tsujii, 2007) and deep parsers (Kaplan et al., 2004; Clark and Curran, 2004; Miyao and Tsujii, 2008).
---------------------------------------------------
P08-1006:85	35:181	NO-RERANK Charniak (2000)s parser, based on a lexicalized PCFG model of phrase structure trees.3 The probabilities of CFG rules are parameterized on carefully hand-tuned extensive information such as lexical heads and symbols of ancestor/sibling nodes.
---------------------------------------------------
D08-1052:86	20:235	The designs of hypotheses in (Collins, 1999; Charniak, 2000) show a delicate balance between expressiveness and tractability, which play an important role in natural language parsing.
---------------------------------------------------
W07-2028:87	17:85	A tree kernel (Moschitti, 2004) is used to exploit the deep syntactic processing obtained using the Charniak parser (Charniak, 2000).
---------------------------------------------------
P04-2009:88	83:110	5.1 Parsers used Charniaks parser (2000) is a combination probabilistic context free grammar and maximum entropy parser.
---------------------------------------------------
W08-0401:89	140:232	For extraction of source sentence tree structure, we used the Charniak parser (Charniak 2000).
---------------------------------------------------
J06-2003:90	514:682	In order to use this tool, we parsed the English sentences with Charniaks parser (Charniak 2000).
---------------------------------------------------
N06-1023:91	28:177	Recently, many accurate statistical parsers have been proposed (e.g. , (Collins, 1999; Charniak, 2000) for English, (Uchimoto et al. , 2000; Kudo and Matsumoto, 2002) for Japanese).
---------------------------------------------------
N06-1023:92	11:177	Blaheta and Charniak proposed a statistical method Currently, National Institute of Information and Communications Technology, JAPAN, dk@nict.go.jp Currently, Graduate School of Informatics, Kyoto University, kuro@i.kyoto-u.ac.jp for analyzing function tags in Penn Treebank, and achieved a really high accuracy of 95.7% for syntactic roles, such as SBJ (subject) and DTV (dative) (Blaheta and Charniak, 2000).
---------------------------------------------------
J03-4003:93	675:797	Charniak (2000) describes a parsing model that also uses Markov processes to generate rules.
---------------------------------------------------
J03-4003:94	682:797	Finally, Bod (2001) describes a very different approach (a DOP approach to parsing) that gives excellent results on treebank parsing, comparable to the results of Charniak (2000) and Collins (2000).
---------------------------------------------------
J03-4003:95	713:797	Charniak (2000) shows that using the POS tags as word class information in the model is important for parsing accuracy.
---------------------------------------------------
J03-4003:96	343:797	Two models (Collins 2000; Charniak 2000) outperform models 2 and 3 on section 23 of the treebank.
---------------------------------------------------
J03-4003:97	667:797	Blaheta and Charniak (2000) describe methods for the recovery of the semantic tags in the Penn Treebank annotations, a significant step forward from the complement/adjunct distinction recovered in model 2 of the current article.
---------------------------------------------------
J03-4003:98	345:797	Charniak (2000) describes a series of enhancements to the earlier model of Charniak (1997).
---------------------------------------------------
D07-1078:99	19:178	Binarizing the syntax trees for syntax-based machine translation is similar in spirit to generalizing parsing models via markovization (Collins, 1997; Charniak, 2000).
---------------------------------------------------
J06-1005:100	114:684	7 Besides the work on semantic roles (Charniak 2000; Gildea and Jurafsky 2002; Thompson, Levy, and Manning 2003), considerable interest has been shown in the automatic interpretation of various noun phrase-level constructions, such as noun compounds.
---------------------------------------------------
J06-1005:101	281:684	Each sentence in this corpus was then parsed using the syntactic parser developed by Charniak (2000).
---------------------------------------------------
W03-1017:102	67:165	Syntactic structure was obtained with Charniaks statistical parser (Charniak, 2000).
---------------------------------------------------
W02-1039:103	43:151	When an S alignment exists, there will always also exist a P alignment such that P a65 S. The English sentences were parsed using a state-of-the-art statistical parser (Charniak, 2000) trained on the University of Pennsylvania Treebank (Marcus et al. , 1993).
---------------------------------------------------
W02-1006:104	66:163	1(a) attention (noun) 1(b) He turned his attention to the workbench . 1(c) a24 turned, VBD, active, lefta41 2(a) turned (verb) 2(b) He turned his attention to the workbench . 2(c) a24 he, attention, PRP, NN, VBD, activea41 3(a) green (adj) 3(b) The modern tram is a green machine . 3(c) a24 machine, NNa41 Table 1: Examples of syntactic relations (assuming no feature selection) 3.4 Syntactic Relations We first parse the sentence containing a2 with a statistical parser (Charniak, 2000).
---------------------------------------------------
P05-1073:105	165:185	We trained and tested on automatic parse trees from Charniaks parser (Charniak, 2000).
---------------------------------------------------
P05-1073:106	15:185	We present performance results on the February 2004 version of PropBank on gold-standard parse trees as well as results on automatic parses generated by Charniaks parser (Charniak, 2000).
---------------------------------------------------
P09-2026:107	26:88	The system uses as input the paired corpus, the corresponding POS tagged corpus, the paired corpus parsed using the Charniak parser (Charniak, 2000), and dependency parses from the MST parser (McDonald et al., 2005).
---------------------------------------------------
W04-0307:108	128:154	5.2 Comparing to Other Parsers Charniaks state-of-the-art PCFG parser (Charniak, 2000) has achieved the highest PARSEVAL LP/LR when compared to Collins Model 2 and Model 3 (Collins, 1999), Roarks (Roark, 2001), Ratnaparkhis (Ratnaparkhi, 1999), and Xu & Chelbas (Xu et al. , 2002) parsers.
---------------------------------------------------
W04-0307:109	5:154	1 Introduction Statistical parsing has been an important focus of recent research (Magerman, 1995; Eisner, 1996; Charniak, 1997; Collins, 1999; Ratnaparkhi, 1999; Charniak, 2000).
---------------------------------------------------
W04-0307:110	9:154	Charniak (Charniak, 2000) developed a state-of-the-art statistical CFG parser and then built an effective language model based on it (Charniak, 2001).
---------------------------------------------------
W04-0307:111	145:154	Models  40 words (2,245 sentences) Without TRACE All (1,903 sentences) (2,245 sentences) governor only all roles governor only all roles RLP RLR RLP RLR RLP RLR RLP RLR L 92.4 92.4 89.5 88.7 92.4 92.3 89.1 88.6 T 93.2 92.9 89.9 89.3 93.2 92.9 89.8 89.2 Charniak (Charniak, 2000) 92.6 92.5 89.4 88.9 92.5 92.3 88.9 88.7 Collins, Model 2 (Collins, 1999) 92.5 92.3 89.1 88.5 92.2 92.1 89.0 88.5 Collins, Model 3 (Collins, 1999) 92.8 92.7 89.9 89.4 92.7 92.4 89.3 89.1 Models  100 words (2,416 sentences) Without TRACE All (1,979 sentences) (2,416 sentences) governor only all roles governor only all roles RLP RLR RLP RLR RLP RLR RLP RLR L 91.9 91.6 88.8 88.1 91.8 91.5 88.5 87.8 T 92.7 92.3 89.4 88.7 92.6 92.2 89.1 88.5 Charniak (Charniak, 2000) 92.0 91.8 88.8 88.2 91.9 91.6 88.4 87.9 Collins, Model 2 (Collins, 1999) 91.8 91.6 88.6 88.0 91.7 91.5 88.2 87.9 Collins, Model 3 (Collins, 1999) 92.2 92.1 89.4 88.8 92.1 91.9 88.8 88.5 CFG parsers may loose accuracy from the CFG-toCDG transformation, similarly to Collins experiment reported in (Hajic et al. , 1998), we also transformed our CDG parses to Penn Treebank style CFG parse trees and scored them using PARSEVAL.
---------------------------------------------------
W04-0307:112	96:154	As in (Ratnaparkhi, 1999; Charniak, 2000; Collins, 1999), we evaluate on all sentences with length  40 words (2,245 sentences) and length  100 words (2,416 sentences).
---------------------------------------------------
W04-0307:113	6:154	Several of these parsers generate constituents by conditioning probabilities on non-terminal labels, part-of-speech (POS) tags, and some headword information (Collins, 1999; Ratnaparkhi, 1999; Charniak, 2000).
---------------------------------------------------
W07-2053:114	66:89	We use MXPOST tagger (Adwait, 1996) for POS tagging, Charniak parser (Charniak, 2000) for extracting syntactic relations, and David Blei?s version of LDA1 for LDA training and inference.
---------------------------------------------------
D07-1108:115	103:175	We use MXPOST tagger (Adwait, 1996) for POS tagging, Charniak parser (Charniak, 2000) for extracting syntactic relations, SVMlight1 for SVM classifier and David Bleis version of LDA2 for LDA training and inference.
---------------------------------------------------
W06-2903:116	7:150	Compared to a basic treebank grammar (Charniak, 1996), the grammars of highaccuracy parsers weaken independence assumptions by splitting grammar symbols and rules with either lexical (Charniak, 2000; Collins, 1999) or nonlexical (Klein and Manning, 2003; Matsuzaki et al. , 2005) conditioning information.
---------------------------------------------------
D07-1022:117	9:273	Take, for example, parsing the Wall Street Journal Penn Treebank, a longstanding task for which highly accurate context-free models stabilized by the year 2000 (Collins, 1999; Charniak, 2000).
---------------------------------------------------
W00-1303:118	17:181	Decision Trees(Haruno et al. , 1998) and Maximum Entropy models(Ratnaparkhi, 1997; Uchimoto et al. , 1999; Charniak, 2000) have been applied to dependency or syntactic structure analysis.
---------------------------------------------------
W05-1515:119	219:241	(2004) 89.12 89.10 89.14 89.98 90.22 89.74 Charniak (2000) 90.09 90.01 90.17 89.50 89.69 89.32 bias is to cope with the noise by favoring negative confidences predictions for ambiguous ADJP and ADVP decisions, hence their abysmal labelled recall.
---------------------------------------------------
W05-1515:120	196:241	3.7 Test Set Results Table 9 shows the results of our best parser on the 15 words test set, as well as the accuracy reported for a recent discriminative parser (Taskar et al. , 2004) and scores we obtained by training and testing the parsers of Charniak (2000) and Bikel (2004) on the same data.
---------------------------------------------------
W05-1515:121	198:241	Both Charniak (2000) and Bikel (2004) were trained using the goldstandard tags, as this produced higher accuracy on the development set than using Ratnaparkhi (1996)s tags.
---------------------------------------------------
P03-1054:122	198:233	12This is part of the explanation of why (Charniak, 2000) finds that early generation of head tags as in (Collins, 1999) is so beneficial.
---------------------------------------------------
P03-1054:123	8:233	In the following decade, great success in terms of parse disambiguation and even language modeling was achieved by various lexicalized PCFG models (Magerman, 1995; Charniak, 1997; Collins, 1999; Charniak, 2000; Charniak, 2001).
---------------------------------------------------
P03-1054:124	18:233	Charniak (2000) shows the value his parser gains from parentannotation of nodes, suggesting that this information is at least partly complementary to information derivable from lexicalization, and Collins (1999) uses a range of linguistically motivated and carefully hand-engineered subcategorizations to break down wrong context-freedom assumptions of the naive Penn treebank covering PCFG, such as differentiating base NPs from noun phrases with phrasal modifiers, and distinguishing sentences with empty subjects from those where there is an overt subject NP.
---------------------------------------------------
W04-0308:125	10:145	Parsers that attempt to disambiguate the input completely  full parsing  typically first employ some kind of dynamic programming algorithm to derive a packed parse forest and then applies a probabilistic top-down model in order to select the most probable analysis (Collins, 1997; Charniak, 2000).
---------------------------------------------------
P07-1080:126	143:171	It should also be noted that the model (Charniak, 2000) is the most accurate generative model on the standard WSJ parsing benchmark, which confirms the viability of our generative model.
---------------------------------------------------
P07-1080:127	140:171	637 ferent generative and discriminative parsing methods (Bikel, 2004; Taskar et al. , 2004; Turian and Melamed, 2006; Charniak, 2000) evaluated in the same experimental setup.
---------------------------------------------------
P07-1080:128	142:171	This improvement is statically significant.3 The MF model achieves results which do not appear to be significantly different from the results of the best model in the list (Charniak, 2000).
---------------------------------------------------
P07-1080:129	17:171	(Charniak, 2000; Collins, 1999)).
---------------------------------------------------
P07-1080:130	68:171	Although the most accurate parsing models (Charniak and Johnson, 2005; Henderson, 2004; Collins, 2000) are discriminative, all the most accurate discriminative models make use of a generative model.
---------------------------------------------------
P07-1080:131	126:171	It is expensive to train R P F1 Bikel, 2004 87.9 88.8 88.3 Taskar et al. , 2004 89.1 89.1 89.1 NN method 89.1 89.2 89.1 Turian and Melamed, 2006 89.3 89.6 89.4 MF method 89.3 90.7 90.0 Charniak, 2000 90.0 90.2 90.1 Table 1: Percentage labeled constituent recall (R), precision (P), combination of both (F1) on the testing set.
---------------------------------------------------
P07-1080:132	164:171	Both methods are empirically compared, and the mean field approach achieves significantly better results, which are non-significantly different from the results of the most accurate generative parsing model (Charniak, 2000) on our testing set.
---------------------------------------------------
W03-0312:133	58:158	The Japanese parser outputs the phrasal dependency structure of an input, and that is used as is. We used The Japanese parser KNP (Kurohashi and Nagao, 1994) and The English nl-parser (Charniak, 2000).
---------------------------------------------------
N06-2026:134	72:89	These systems all use (Charniak, 2000)s parse trees both for training and testing, as well as various other information sources including sets of n-best parse trees, chunks, or named entities.
---------------------------------------------------
N06-2026:135	4:89	1 Introduction Recent successes in statistical syntactic parsing based on supervised techniques trained on a large corpus of syntactic trees (Collins, 1999; Charniak, 2000; Henderson, 2003) have brought the hope that the same approach could be applied to the more ambitious goal of recovering the propositional content and the frame semantics of a sentence.
---------------------------------------------------
N06-2026:136	68:89	However, state-of-theart semantic role labelling systems (CoNLL, 2005) use parse trees output by state-of-the-art parsers (Collins, 1999; Charniak, 2000), both for training and testing, and return partial trees annotated with semantic role labels.
---------------------------------------------------
D09-1108:137	200:261	We use FBIS corpus (250K sentence pairs) as training data with the source side parsed by a modified Charniak parser (Charniak 2000) which can output a packed forest.
---------------------------------------------------
H05-1001:138	142:277	4.4 Results with GUITAR To use GUITAR, we first parsed the texts using Charniaks parser (Charniak, 2000).
---------------------------------------------------
N06-1022:139	61:188	(2005) have implemented a dependency parser with good accuracy (it is almost as good at dependency parsing as Charniak (2000)) and very impressive speed (it is about ten times faster than Collins (1997) and four times faster than Charniak (2000)).
---------------------------------------------------
N06-1022:140	42:188	The parser of Charniak (2000) is also a two-stage ctf model, where the first stage is a smoothed Markov grammar (it uses up to three previous constituents as context), and the second stage is a lexicalized Markov grammar with extra annotations about parents and grandparents.
---------------------------------------------------
N06-1022:141	162:188	It has been repeatedly shown to improve parsing accuracy (Johnson, 1998; Charniak, 2000; Klein and Manning, 2003b), but it is difficult if not impossible to integrate with best-first search in bottom-up chart-parsing (as in Charniak et al.
---------------------------------------------------
P03-2012:142	66:175	The texts were parsed by E. Charniaks parser (Charniak, 2000).
---------------------------------------------------
P06-1055:143	8:234	1 Introduction Probabilistic context-free grammars (PCFGs) underlie most high-performance parsers in one way or another (Collins, 1999; Charniak, 2000; Charniak and Johnson, 2005).
---------------------------------------------------
P06-1055:144	11:234	Therefore, a variety of techniques have been developed to both enrich and generalize the naive grammar, ranging from simple tree annotation and symbol splitting (Johnson, 1998; Klein and Manning, 2003) to full lexicalization and intricate smoothing (Collins, 1999; Charniak, 2000).
---------------------------------------------------
P07-1035:145	18:171	Hence, state-of-the-art parsers either supplement the part-of-speech (POS) tags with the lexical forms themselves (Collins, 2003; Charniak, 2000), manually split the tagset into a finer-grained one (Klein and Manning, 2003a), or learn finer grained tag distinctions using a heuristic learning procedure (Petrov et al. , 2006).
---------------------------------------------------
C08-1050:146	55:214	A constituent-based system using Charniaks parser (Charniak, 2000).
---------------------------------------------------
C08-1050:147	16:214	By habit, most systems for automatic role-semantic analysis have used Pennstyle constituents (Marcus et al., 1993) produced by Collins (1997) or Charniaks (2000) parsers.
---------------------------------------------------
E09-1092:148	28:181	((:I (:Q DET NAMED-ENTITY) ENTER[V] (:Q THE ROOM[N])) (:I (:Q DET FEMALE-INDIVIDUAL) HAVE[V] (:Q DET ROOM[N])) (:I (:Q DET FEMALE-INDIVIDUAL) SLEEP[V]) (:I (:Q DET FEMALE-INDIVIDUAL) HAVE[V] (:Q DET (:F PLUR CLOTHE[N]))) (:I (:Q DET (:F PLUR CLOTHE[N])) WASHED[A])) Here the upper-case sentences are automatically generated verbalizations of the abstracted LFs shown beneath them.1 The initial development of KNEXT was based on the hand-constructed parse trees in the Penn Treebank version of the Brown corpus, but subsequently Schubert and collaborators refined and extended the system to work with parse trees obtained with statistical parsers (e.g., that of Collins (1997) or Charniak (2000)) applied to larger corpora, such as the British National Corpus (BNC), a 100 million-word, mixed genre collection, along with Web corpora of comparable size (see work of Van Durme et al.
---------------------------------------------------
C04-1021:149	75:219	2.1 Parser Enhancements We used the Charniak parser (Charniak, 2000) for the experiments reported in this paper.
---------------------------------------------------
P06-2089:150	9:169	Some of the more popular and more accurate of these approaches to data-driven parsing (Charniak, 2000; Collins, 1997; Klein and Manning, 2002) have been based on generative models that are closely related to probabilistic contextfree grammars.
---------------------------------------------------
P06-2089:151	148:169	We tested this hypothesis by using the Charniak (2000) parser in n-best mode, producing the top 10 trees with corresponding probabilities.
---------------------------------------------------
P06-2089:152	141:169	In the probabilistic LR model, probabilities are assigned to tree 696 Precision Recall F-score Time (min) Best-First Classifier-Based (this paper) 88.1 87.8 87.9 17 Deterministic (MaxEnt) (this paper) 85.4 84.8 85.1 < 1 Charniak & Johnson (2005) 91.3 90.6 91.0 Unk Bod (2003) 90.8 90.7 90.7 145* Charniak (2000) 89.5 89.6 89.5 23 Collins (1999) 88.3 88.1 88.2 39 Ratnaparkhi (1997) 87.5 86.3 86.9 Unk Tsuruoka & Tsujii (2005): deterministic 86.5 81.2 83.8 < 1* Tsuruoka & Tsujii (2005): search 86.8 85.0 85.9 2* Sagae & Lavie (2005) 86.0 86.1 86.0 11* Table 1: Summary of results on labeled precision and recall of constituents, and time required to parse the test set.
---------------------------------------------------
P06-2089:153	153:169	See (Charniak, 2000) for details.
---------------------------------------------------
P06-2089:154	140:169	A probabilistic shiftreduce LR-like model, such as the one used in our parser, is different in many ways from a lexicalized PCFG-like model (using markov a grammar), such as those used in the Collins (1999) and Charniak (2000) parsers.
---------------------------------------------------
W08-0406:155	106:250	The English side is parsed using a state-of-the-art statistical English parser (Charniak, 2000).
---------------------------------------------------
W08-1005:156	6:154	1 Introduction Probabilistic context-free grammars (PCFGs) underlie most high-performance parsers in one way or another (Collins, 1999; Charniak, 2000; Charniak and Johnson, 2005).
---------------------------------------------------
W08-1005:157	9:154	Therefore, a variety of techniques have been developed to both enrich and generalize the naive grammar, ranging from simple tree annotation and symbol splitting (Johnson, 1998; Klein and Manning, 2003) to full lexicalization and intricate smoothing (Collins, 1999; Charniak, 2000).
---------------------------------------------------
D09-1012:158	99:201	and automatic Charniak parse trees (Charniak, 2000) as provided for the CoNLL 2005 evaluation campaign (Carreras and M`arquez, 2005).
---------------------------------------------------
P06-1022:159	87:186	Incidentally, the above attributes are the same as those used by the conventional stochastic dependency parsing methods (Collins, 1996; Ratnaparkhi, 1997; Fujio and Matsumoto, 1998; Uchimoto et al. , 1999; Charniak, 2000; Kudo and Matsumoto, 2002).
---------------------------------------------------
E06-1011:160	48:205	The score of a tree for secondorder parsing is now s(x,y) = summationdisplay (i,k,j)y s(i,k,j) where k and j are adjacent, same-side children of i in the tree y. The second-order model allows us to condition onthe mostrecent parsing decision, thatis, the last dependent picked up by a particular word, which is analogous to the the Markov conditioning of in the Charniak parser (Charniak, 2000).
---------------------------------------------------
E06-1011:161	18:205	Previous work has shown that conditioning on neighboring decisions can lead to significant improvements in accuracy (Yamada and Matsumoto, 2003; Charniak, 2000).
---------------------------------------------------
W07-1527:162	84:168	The resulting English corpus contained 10,000 sentences which were syntactically parsed (Charniak, 2000).
---------------------------------------------------
C04-1040:163	22:230	However, the accuracy of Yamadas analyzer is lower than state-of-the-art phrase structure parsers such as Charniaks Maximum-Entropy-Inspired Parser (MEIP) (Charniak, 2000) and Collins Model 3 parser.
---------------------------------------------------
P09-1022:164	73:180	(2008), the version of 192 the LFG parser which applies the LFG annotation algorithm to the earlier Charniaks parser (Charniak, 2000) obtains an f-score of 86.97 on the Wall Street Journal Section 23 test set.
---------------------------------------------------
W06-1603:165	75:249	A pair of sentences is rst fed to a syntactic parser (Charniak, 2000) and then passed to a semantic role labeler (ASSERT; (Pradhan et al. , 2004)), to label predicate argument tuples.
---------------------------------------------------
P03-1069:166	97:237	The corpus is distributed in a Treebankstyle machine-parsed version which was produced with Charniaks (2000) parser.
---------------------------------------------------
D09-1085:167	105:198	The Stanford parser is representative of a large number of PTB parsers, exemplified by Collins (1997) and Charniak (2000).
---------------------------------------------------
P06-1033:168	55:272	Thus, Nivre and Nilsson (2005) improve parsing accuracy for MaltParser by projectivizing training data and applying an inverse transformation to the output of the parser, while Hall and Novak (2005) apply post-processing to the output of Charniaks parser (Charniak, 2000).
---------------------------------------------------
J07-3004:169	74:503	In this, they differ crucially from the bilexical surface dependencies used by the parsing models of Collins (1999) and Charniak (2000) and returned by the dependency parser of McDonald, Crammer, and Pereira (2005).
---------------------------------------------------
J07-3004:170	26:503	Although there is now a sizable literature on trace and function-tag insertion algorithms (Blaheta and Charniak 2000; Johnson 2002; Campbell 2004), and integrated parsing with function tags or null elements (Dienes and Dubey 2003a, 2003b; Merlo and Musillo 2005; Gabbard, Kulick, and Marcus 2006), such approaches typically require additional preor postprocessing steps that are likely to add further noise and errors to the parser output.
---------------------------------------------------
J07-3004:171	19:503	Statistical parsing on the Penn Treebank has made great progress by focusing on the machine-learning or algorithmic aspects (Magerman 1994; Ratnaparkhi 1998; Collins 1999; Charniak 2000; Henderson 2004; McDonald, Crammer, and Pereira 2005).
---------------------------------------------------
J07-3004:172	285:503	The dependencies involved in these constructions, however, are difficult to obtain from the output of standard parsers such as Collins (1999) or Charniak (2000), and require additional postprocessing that may introduce further noise and errors.
---------------------------------------------------
J07-3004:173	473:503	Unlike most Penn Treebank parsers, such as Collins (1999) or Charniak (2000), these CCGbank parsers return not only syntactic derivations, but also local and long-range dependencies, includ390 Hockenmaier and Steedman CCGbank ing those that arise under relativization and coordination.
---------------------------------------------------
W06-2611:174	173:216	The sentences were processed using Charniaks parser (Charniak, 2000) to generate parse trees automatically.
---------------------------------------------------
P07-2052:175	7:109	1 Introduction In recent years, many accurate phrase-structure parsers have been developed (e.g. , (Collins, 1999; Charniak, 2000)).
---------------------------------------------------
I08-1016:176	57:147	In our implementation, a context free grammar probabilistic parser (Charniak, 2000) was used to parse the input.
---------------------------------------------------
P07-1025:177	85:217	For this task we utilized the August 2005 release of the Charniak parser with the default speed/accuracy settings (Charniak, 2000), which required roughly 360 hours of processor time on a 2.5 GHz PowerPC G5.
---------------------------------------------------
W03-1707:178	12:178	The creation of the Penn English Treebank (Marcus et al. , 1993), a syntactically interpreted corpus, played a crucial role in the advances in natural language parsing technology (Collins, 1997; Collins, 2000; Charniak, 2000) for English.
---------------------------------------------------
D08-1091:179	11:223	In recent years, latent variable methods have been shown to produce grammars which are as good as, or even better than, earlier parsing work (Collins, 1999; Charniak, 2000).
---------------------------------------------------
W06-1622:180	60:165	It includes several Wall Street Journal sections with parse-trees from both Charniaks (2000) parser and Collins (1999) parser.
---------------------------------------------------
W04-0305:181	71:189	3 A Generative Left-Corner Probability Model As with several previous statistical parsers (Collins, 1999; Charniak, 2000), we use a generative history-based probability model of parsing.
---------------------------------------------------
P03-1046:182	51:174	Therefore, the head words of arguments (such as Smith) are generated in the following manner: C8B4DB CP CYCR CP BNCWCWCR CW BNDB CW CXBNCXBNCWCR CP BNDB CP CXCXB5 The head word of modifiers (such as yesterday)are generated differently: C8B4DB D1 CYCR D1 BNCWCWCR D1 BNDB D1 CXBNCXBNCWCR CW BNDB CW CXB5 Like Collins (1999) and Charniak (2000), the SD model assumes that word-word dependencies can be defined at the maximal projection of a constituent.
---------------------------------------------------
P03-1046:183	4:174	1 Introduction State-of-the-art statistical parsers for Penn Treebank-style phrase-structure grammars (Collins, 1999), (Charniak, 2000), but also for Categorial Grammar (Hockenmaier and Steedman, 2002b), include models of bilexical dependencies defined in terms of local trees.
---------------------------------------------------
N04-2009:184	27:135	The documents are then parsed using Eugene Charniaks maximum entropy inspired parser (Charniak, 2000).
---------------------------------------------------
P07-1122:185	36:184	(1999) and Charniak (2000) adapted to Czech are not able to create the non-projective arcs present in the treebank, which is unsatisfactory.
---------------------------------------------------
P07-1122:186	7:184	This can be seen in state-of-the-art constituency-based parsers such as Collins (1999), Charniak (2000), and Petrovetal.(2006),andtheeffectsofdifferenttransformations have been studied by Johnson (1998), KleinandManning(2003),andBikel(2004).
---------------------------------------------------
W09-1008:187	66:224	Solving this second methodological issue has led to solutions dubbed hereafter as lexicalized parsing (Charniak, 2000; Collins, 1999).
---------------------------------------------------
P04-1042:188	97:176	In Table 2 we present comparative results, using the PARSEVAL-based evaluation metric introduced by Johnson (2002)  a correct empty category inference requires the string position of the empty category, combined with the left and right boundaries plus syntactic category of the antecedent, if any, for purposes of comparison.9,10 Note that this evaluation metric does not require correct attachment of the empty category into 8A complete description of feature templates can be found at http://nlp.stanford.edu/rog/acl2004/templates/index.html 9For purposes of comparability with Johnson (2002) we used Charniaks 2000 parser as P. 10Our algorithm was evaluated on a more stringent standard for NP-* than in previous work: control loci-related mappings were done after dislocated nodes were actually relocated by the algorithm, so an incorrect dislocation remapping can render incorrect the indices of a correct NP-* labeled bracketing.
---------------------------------------------------
P04-1042:189	137:176	Note that the dependency figures of Dienes lag behind even the parsed results for Johnsons model; this may well be due to the fact that Dienes built his model as an extension of Collins (1999), which lags behind Charniak (2000) by about 1.3-1.5%.
---------------------------------------------------
P04-1042:190	126:176	State-of-the-art statistical parsing is far better on WSJ (Charniak, 2000) than on NEGRA (Dubey and Keller, 2003), so for comparison of parser-composed dependency performance we used vanilla PCFG models for both WSJ and NEGRA trained on comparably-sized datasets; in addition to making similar types of independence assumptions, these models performed relatively comparably on labeled bracketing measures for our development sets (73.2% performance for WSJ versus 70.9% for NEGRA).
---------------------------------------------------
W08-2226:191	22:118	TEXTCAP first uses the domain-independent Charniak parser (Charniak, 2000) to convert sentences in the source document into a sequence of syntactic parses.
---------------------------------------------------
W08-2226:192	30:118	For instance, the introductionof large-scalelexicaland syntacticresources likethePennTreeBank(Marcusetal.,1993)haveledto highlyaccurate,domainindependent parsers (Collins, 1999; Charniak, 2000).
---------------------------------------------------
N09-3017:193	38:131	used to describe a quality apparent in a person We parsed the sentences using the Charniak parser (Charniak, 2000).
---------------------------------------------------
P09-1020:194	208:277	We train Charniaks parser (Charniak 2000) on CTB5 to do Chinese parsing, and modify it to output packed forest.
---------------------------------------------------
C02-1159:195	23:252	Using existing parsers such as (Charniak, 2000; Collins, 1999) would eliminate the need to build a parser from scratch.
---------------------------------------------------
W06-1668:196	11:239	Additionally, many discriminative models use a generative model as a base model and add discriminative features with reranking (Collins, 2000; Charniak and Johnson, 2005; Roark et al. , 2004), or train discriminatively a small set of weights for features which are generatively estimated probabilities (Raina et al. , 2004; Och and Ney, 2002).
---------------------------------------------------
W06-1668:197	14:239	To avoid this problem, generative models for NLP tasks have often been manually designed to achieve an appropriate representation of the joint distribution, such as in the parsing models of (Collins, 1997; Charniak, 2000).
---------------------------------------------------
D07-1072:198	25:303	Lexical methods split each pre-terminal symbol into many subsymbols, one for each word, and then focus on smoothing sparse lexical statis688 tics (Collins, 1999; Charniak, 2000).
---------------------------------------------------
C00-1011:199	126:160	40,000 sentences) and section 23 for testing (see Collins 1997, 1999; Charniak 1997, 2000; l~,atnalmrkhi 1999); we only tested on sentences _< 40 words (2245 sentences).
---------------------------------------------------
C00-1011:200	142:160	These scores are higher than those of several other parsers (e.g. Collins 1997, 99; Charniak 1997), but remain behind tim scores of Charniak (2000) who obtains 90.1% LP and 90.1% LR for sentences _< 40 words.
---------------------------------------------------
W05-0623:201	67:79	Recent releases of the Charniak parser (Charniak, 2000) have included an option to provide the top k parses of a given sentence according to the probability model of the parser.
---------------------------------------------------
C04-1097:202	52:154	(24 for German, 23 for French) We denote that set of features in shorthand as () i f  . With this extension, a model of Markov 3 A Markov grammar is a model of constituent structure that starts at the root of the tree and assigns probability to the expansion of a non-terminal one daughter at a time, rather than as entire productions (Charniak, 1997 & 2000).
---------------------------------------------------
C04-1097:203	150:154	Experiments by Daum et al (2002) and the parsing work of Charniak (2000) and others indicate that further lexicalization may yield some additional improvements for ordering.
---------------------------------------------------
C04-1097:204	124:154	The PTB to DSIF transformation pipeline includes the following stages, inspired by Langkilde-Gearys (2002b) description: A. Deserialize the tree B. Label heads, according to Charniaks head labeling rules (Charniak, 2000) C. Remove empty nodes and flatten any remaining empty non-terminals D. Relabel heads to conform more closely to the head conventions of NLPWin E. Label with logical roles, inferred from PTB functional roles F. Flatten to maximal projections of heads (MPH), except in the case of conjunctions G. Flatten non-branching non-terminals H. Perform dictionary look-up and morphological analysis I. Introduce structure for material between paired delimiters and for any coordination not already represented in the PTB J. Remove punctuation K. Remove function words L. Map the head of each maximal projection to a dependency node, and map the heads modifiers to the first nodes dependents, thereby forming a complete dependency tree.
---------------------------------------------------
J04-3001:205	10:405	Current state-of-the-art statistical parsers (Collins 1999; Charniak 2000) are all trained on large annotated corpora such as the Penn Treebank (Marcus, Santorini, and Marcinkiewicz 1993).
---------------------------------------------------
W07-2085:206	62:108	(Charniak, 2000)).
---------------------------------------------------
W00-1201:207	7:100	The success of statistical methods in particular has been quite evident in the area of syntactic parsing, most recently with the outstanding results of (Charniak, 2000) and (Colhns, 2000) on the now-standard English test set of the Penn Treebank (Marcus et al. , 1993).
---------------------------------------------------
P08-1013:208	7:210	More accurate statistical models of natural language have mainly been developed in the field of statistical parsing, e.g. Collins (2003), Charniak (2000) and Ratnaparkhi (1999).
---------------------------------------------------
W06-2002:209	4:162	1 Introduction Much of the current research into probabilistic parsing is founded on probabilistic contextfreegrammars(PCFGs)(Collins,1999;Charniak, 2000; Charniak, 2001).
---------------------------------------------------
N06-2019:210	24:84	Experiments performed (3.3) use a state-of-the-art parser (Charniak, 2000) to study the impact of early filler deletion under in-domain and out-of-domain (i.e. adaptation) training conditions.
---------------------------------------------------
N06-2019:211	5:84	Reported experiments measure the effect of early deletion under in-domain and out-of-domain parser training conditions using a state-of-the-art parser (Charniak, 2000).
---------------------------------------------------
N06-2019:212	50:84	The dependency metric performs syntactic head-matching for each word using a set of given head percolation rules (derived from Charniaks parser (2000)), and its relaxed formulation ignores terminals spanned by FILLER and EDITED constituents.
---------------------------------------------------
N06-2019:213	33:84	3 Experiments This section reports parsing experiments studying the effect of early deletion under in-domain and outof-domain parser training conditions using the August 2005 release of the Charniak parser (2000).
---------------------------------------------------
W06-2305:214	12:167	Compared to traditional handwritten grammars, successfulstochasticmodelslike(Collins,1999;Charniak, 2000) open up an even greater space of alternatives for the parser and accordingly offer a great deal of opportunities to construct odd structural descriptions from them.
---------------------------------------------------
P08-1037:215	30:210	Parsing As our baseline parsers, we use two state-of-theart lexicalised parsing models, namely the Bikel parser (Bikel, 2004) and Charniak parser (Charniak, 2000).
---------------------------------------------------
P08-1037:216	9:210	For example, a number of different parsers have been shown to benet from lexicalisation, that is, the conditioning of structural features on the lexical head of the given constituent (Magerman, 1995; Collins, 1996; Charniak, 1997; Charniak, 2000; Collins, 2003).
---------------------------------------------------
P08-1067:217	130:181	Our feature set is summarized in Table 2, which closely follows Charniak and Johnson (2005), except that we excluded the non-local features Edges, NGram, and CoPar, and simplified Rule and NGramTree features, since they were too complicated to compute.4 We also added four unlexicalized local features from Collins (2000) to cope with data-sparsity.
---------------------------------------------------
P08-1067:218	167:181	type system F1% D Collins (2000) 89.7 Henderson (2004) 90.1 Charniak and Johnson (2005) 91.0 updated (Johnson, 2006) 91.4 this work 91.7 G Bod (2003) 90.7Petrov and Klein (2007) 90.1 S McClosky et al.
---------------------------------------------------
P08-1067:219	47:181	Following (Charniak and Johnson, 2005), the first feature f1(y) = logPr(y) is the log probability of a parse from the baseline generative parser, while the remaining features are all integer valued, and each of them counts the number of times that a particular configuration occurs in parse y. For example, one such feature f2000 might be a question how many times is a VP of length 5 surrounded by the word has and the period?
---------------------------------------------------
P08-1067:220	49:181	Using a machine learning algorithm, the weight vector w can be estimated from the training data where each sentence si is labelled with its correct (gold-standard) parse yi . As for the learner, Collins (2000) uses the boosting algorithm and Charniak and Johnson (2005) use the maximum entropy estimator.
---------------------------------------------------
P08-1067:221	39:181	However, in this work, we use forests from a Treebank parser (Charniak, 2000) whose grammar is often flat in many productions.
---------------------------------------------------
P08-1067:222	125:181	Those with a  are from (Collins, 2000), and others are from (Charniak and Johnson, 2005), with simplifications.
---------------------------------------------------
C02-1126:223	23:139	criminative models described in (Magerman, 1995; Ratnaparkhi, 1997), the lexicalized PCFG models in (Collins, 1999), the generative model in (Charniak, 2000), the lexicalized TAG extractor in (Xia, 1999) and the stochastic lexicalized TAG models in (Chiang, 2000; Sarkar, 2001; Chen and VijayShanker, 2000).
---------------------------------------------------
P06-2010:224	127:184	Train Devel tWSJ tBrown Sentences 39,832 1,346 2,416 426 Tokens 950,028 32,853 56,684 7,159 Propositions 90,750 3,248 5,267 804 Arguments 239,858 8,346 14,077 2,177 Table 2: Counts on the data set The preprocessing modules used in CONLL2005 include an SVM based POS tagger (Gimenez and M`arquez, 2003), Charniak (2000)s full syntactic parser, and Chieu and Ng (2003)s Named Entity recognizer.
---------------------------------------------------
P05-1025:225	4:182	By using a statistical parser (Charniak, 2000) and memorybased learning tools for classification (Daelemans et al. , 2004), we obtain high precision and recall of several GRs.
---------------------------------------------------
P05-1025:226	50:182	This dependency extraction procedure from constituent trees gives us a straightforward way to obtain unlabeled dependencies: use an existing statistical parser (Charniak, 2000) trained on the Penn Treebank to produce constituent trees, and extract unlabeled dependencies using the aforementioned head-finding rules.
---------------------------------------------------
P07-1078:227	8:197	1 Introduction State of the art statistical parsers (Collins, 1999; Charniak, 2000; Koo and Collins, 2005; Charniak and Johnson, 2005) are trained on manually annotated treebanks that are highly expensive to create.
---------------------------------------------------
C04-1157:228	92:133	5.1 Parsers used Charniaks parser (2000) is a combination probabilistic context free grammar and maximum entropy parser.
---------------------------------------------------
P06-3004:229	10:199	For the English WSJ, high accuracy parsing models have been created, some of them using extensions to classical PCFG parsing such as lexicalization and markovization (Collins, 1999; Charniak, 2000; Klein and Manning, 2003).
---------------------------------------------------
W08-0308:230	125:159	The translation is from English to Chinese, and Charniak (2000)s parser, trained on the Penn Treebank, is used to generate the syntax trees for the English side.
---------------------------------------------------
P04-1030:231	215:232	We achieve 73.2/76.5% LP/LR on section 23 of the Penn Treebank, compared to 82.9/82.4% LP/LR of Roark (2001) and 90.1/90.1% LP/LR of Charniak (2000).
---------------------------------------------------
P04-1030:232	197:232	The current best-performing models, in terms of WER, for the HUB-1 corpus, are the models of Roark (2001), Charniak (2001) (applied to n-best lists by Hall and Johnson (2003)), and the SLM of Chelba and Jelinek (2000) (applied to n-best lists by Xu et al.
---------------------------------------------------
P04-1030:233	204:232	Model n-best List/Lattice Training Size WER (%) SER (%) Oracle (50-best lattice) Lattice 7.8 Charniak (2001) List 40M 11.9 Xu (2002) List 20M 12.3 Roark (2001) (with EM) List 2M 12.7 Hall (2003) Lattice 30M 13.0 Chelba (2000) Lattice 20M 13.0 Current ( a1 1a6 16a0  a1 1) List 20M 13.1 71.0 Current ( a1 1a6 16a0  a1 1) Lattice 20M 13.1 70.4 Roark (2001) (no EM) List 1M 13.4 Lattice Trigram Lattice 40M 13.7 69.0 Current ( a1 1a6 16a0  a1 1) List 1M 14.8 74.3 Current ( a1 1a6 16a0  a1 1) Lattice 1M 14.9 74.0 Current ( a1  a1 0) Lattice 1M 16.0 75.5 Treebank Trigram Lattice 1M 16.5 79.8 No language model Lattice 16.8 84.0 Table 3: Comparison of WER for parsing HUB-1 words lattices with best results of other works.
---------------------------------------------------
E06-1012:234	7:166	With the emergence of the important role of word-to-word relations in parsing (Charniak, 2000; Collins, 1996), dependency grammars have gained acertain popularity; e.g., Yamada and Matsumoto (2003) for English, Kudo and Matsumoto (2000; 2002), Sekine et al.
---------------------------------------------------
P05-1065:235	67:179	We also use a standard statistical parser (Charniak, 2000) to provide syntactic analysis.
---------------------------------------------------
P05-1065:236	105:179	The parse features are generated using the Charniak parser (Charniak, 2000) trained on the standard Wall Street Journal Treebank corpus.
---------------------------------------------------
D07-1065:237	137:188	We evaluate gap insertion on gold trees from section 23 of the Wall Street Journal Penn Treebank (WSJ) and parse trees automatically produced using the Charniak (2000) and Bikel (2004) parsers.
---------------------------------------------------
W06-1623:238	81:250	We automatically segment sentences into clauses using a robust statistical parser (Charniak, 2000).
---------------------------------------------------
D09-1136:239	131:194	4 Experiments We train an English-to-Chinese translation systemusingtheFBIScorpus, where73,597sentence pairs are selected as the training data, and 500 sentencepairswithnomorethan25wordsontheChinese side are selected for both the development and test data.5 Charniak (2000)s parser, trained on the Penn Treebank, is used to generate the English syntax trees.
---------------------------------------------------
D09-1136:240	137:194	5The total 74,597 sentence pairs used in experiments are those in the FBIS corpus whose English part can be parsed using Charniak (2000)s parser.
---------------------------------------------------
W06-2607:241	37:203	2.1 Basic SRL approach The SRL approach that we adopt is based on the deep syntactic parse (Charniak, 2000) of the sentence that we intend to annotate semantically.
---------------------------------------------------
P06-1023:242	172:187	recall f-score this paper 86.0 82.3 84.1 Campbell 85.2 81.7 83.4 Dienes & Dubey 86.5 72.9 79.1 Johnson 85 74 79 Table 5: Accuracy of empty category prediction on section 23 The good performance of our parser on the empty element recognition task is remarkable considering the fact that its performance on the labeled bracketing task is 3% lower than that of the Charniak (2000) parser used by Campbell (2004).
---------------------------------------------------
W06-2303:243	139:147	The partial trees output by these systems were merged with the parse trees returned by (Charniak, 2000)s parser.
---------------------------------------------------
W06-2303:244	3:147	1 Introduction Recent successes in statistical syntactic parsing based on supervised learning techniques trained on a large corpus of syntactic trees (Collins, 1999; Charniak, 2000; Henderson, 2003) have brought forth the hope that the same approaches could be applied to the more ambitious goal of recovering the propositional content and the frame semantics of a sentence.
---------------------------------------------------
W06-2303:245	140:147	These systems use (Charniak, 2000)s parse trees both for training and testing as well as various other information sources including sets of n-best parse trees (Punyakanok et al. , 2005; Haghighi et al. , 2005) or chunks (Marquez et al. , 2005; Pradhan et al. , 2005) and named entities (Surdeanu and Turmo, 2005).
---------------------------------------------------
W06-2303:246	39:147	As with many other statistical parsers (Collins, 1999; Charniak, 2000), SSN parsers use a history-based model of parsing.
---------------------------------------------------
W06-2303:247	135:147	However, state-ofthe-art semantic role labelling systems (CoNLL, 2005) use parse trees output by state-of-the-art parsers (Collins, 1999; Charniak, 2000), both for training and testing, and return partial trees annotated with semantic role labels.
---------------------------------------------------
P06-1006:248	182:193	5.6 Comparison with Different Parsers As mentioned, the above reported results were based on Charniak (2000)s parser.
---------------------------------------------------
P06-1006:249	185:193	We can see that Charniak (2000)s parser leads to higher success rates for NPaper and BNews, while Collins (1999)s achieves better results for NWire.
---------------------------------------------------
P06-1006:250	121:193	The texts were parsed using the maximum-entropybased Charniak parser (Charniak, 2000), based on which the structured features were computed automatically.
---------------------------------------------------
P06-2067:251	8:193	1 Introduction Robust statistical syntactic parsers, made possible by new statistical techniques (Collins, 1999; Charniak, 2000; Bikel, 2004) and by the availability of large, hand-annotated training corpora such as WSJ (Marcus et al. , 1993) and Switchboard (Godefrey et al. , 1992), have had a major impact on the field of natural language processing.
---------------------------------------------------
P06-2067:252	180:193	All these results seem to suggest that adding punctuation in speech transcription is of little help to statistical parsers including at least three state-of-the-art statistical parsers (Collins, 1999; Charniak, 2000; Bikel, 2004).
---------------------------------------------------
W04-2506:253	39:256	To determine which word indicates the semantic class of the expected answer, the syntactic dependencies1 between the question words may be employed (Harabagiu 1Syntactic parsers publicly available, e.g., (Charniak, 2000; et al. , 2000; Pasca and Harabagiu, 2001; Harabagiu et al. , 2001).
---------------------------------------------------
W03-1210:254	88:174	Many researchers ((Blaheta and Charniak 2000), (Gildea and Jurafsky 2000), showed that lexical and syntactic information is very useful for predicateargument recognition tasks, such as semantic roles.
---------------------------------------------------
N06-1039:255	129:236	3.2 Parsing and GLARFing After getting a set of basic clusters, we pass them to an existing statistical parser (Charniak, 2000) and rule-based tree normalizer to obtain a GLARF structure for each sentence in every article.
---------------------------------------------------
P07-1104:256	131:188	The results are presented in Tables 12 and 2, where Baseline results were obtained using the parser by Charniak (2000).
---------------------------------------------------
P07-1104:257	124:188	The linear model combines the log probability calculated by the Charniak (2000) parser as a feature with 1,219,272 additional features.
---------------------------------------------------
P07-1104:258	24:188	For example, in the parse re-ranking task, one cannot tell whether the L2regularized ME approach used by Charniak and Johnson (2005) significantly outperforms the Boosting method by Collins (2000) because different feature sets and n-best parses were used in the evaluations of these methods.
---------------------------------------------------
P07-1104:259	15:188	ME estimators with L2 regularization, which have been widely used in NLP tasks (e.g. , Chen and Rosenfeld 2000; Charniak and Johnson 2005; Johnson et al. 1999), tend to produce models that have this property.
---------------------------------------------------
W05-1529:260	9:45	2 SCF Acquisition System Following the design proposed by Briscoe and Carroll (1997), we built an SCF acquisition system consisting of the following four components: Charniaks parser (Charniak, 2000); an SCF extractor; a lemmatizer; and an SCF evaluator.
---------------------------------------------------
H05-1102:261	196:366	5 Discussion and Future Work The parser proposed in this paper is an incremental parser, so the accuracy on dependency is lower than that for chart parsers, for example like those reported in (Collins, 1999; Charniak, 2000).
---------------------------------------------------
N03-1031:262	6:194	1 Introduction Current state-of-the-art statistical parsers (Collins, 1999; Charniak, 2000) are trained on large annotated corpora such as the Penn Treebank (Marcus et al. , 1993).
---------------------------------------------------
W05-1506:263	42:254	This generalization is not only of theoretical importance, but also critical in the application to state-of-theart parsers such as (Collins, 2003) and (Charniak, 2000).
---------------------------------------------------
W05-1506:264	235:254	60 86 88 90 92 94 96 98 1 2 5 10 20 30 50 70 100 Oracle F-score k (Charniak and Johnson, 2005) This work with beam width 10-4(Collins, 2000) (Ratnaparkhi, 1997) (a) Oracle Reranking 0 2 4 6 8 10 1 2 5 10 20 30 50 70 100 Percentage of Improvement over 1-best k (Charniak and Johnson, 2005) This work with beam width 10-4(Collins, 2000) (Ratnaparkhi, 1997) (b) Relative Improvement Figure 9: Absolutive and Relative F-scores of oracle reranking for the top k ( 100) parses for section 23, compared to (Charniak and Johnson, 2005), (Collins, 2000) and (Ratnaparkhi, 1997).
---------------------------------------------------
W05-1506:265	39:254	They apply this method to the Charniak (2000) parser to get 50-best lists for reranking, yielding an improvement in parsing accuracy.
---------------------------------------------------
W05-1506:266	216:254	We demonstrate this by comparing our k-best lists to those in (Ratnaparkhi, 1997), (Collins, 2000) and the parallel work by Charniak and Johnson (2005) in several ways, including oracle reranking and average number of found parses.
---------------------------------------------------
W05-1506:267	36:254	Since the original design of the algorithm described below, we have become aware of two efforts that are very closely related to ours, one by Jimenez and Marzal (2000) and another done in parallel to ours by Charniak and Johnson (2005).
---------------------------------------------------
W05-1506:268	220:254	Collins (2000), in his parse-reranking experiments, used his Model 2 parser (Collins, 2003) with a beam width of 103 together with a cell limit of 100 to obtain k-best lists; the average number of parses obtained per sentence was 29.2, the maximum, 101.7 Charniak and Johnson (2005) use coarse-tone parsing on top of the Charniak (2000) parser and get 50-best lists for section 23.
---------------------------------------------------
W05-1506:269	191:254	It does support k-best parsing, but, following Collins parse-reranking work (Collins, 2000) (see also Section 5.1.2), it accomplishes this by simply abandoning dynamic programming, i.e., no items are considered equivalent (Charniak and Johnson, 2005).
---------------------------------------------------
P04-1013:270	161:185	For comparison to previous results, table 2 lists the results for our best model (DGSSNFreq 20, rerank)9 and several other statistical parsers (Ratnaparkhi, 1999; Collins, 1999; Collins and Du y, 2002; Charniak, 2000; Collins, 2000; Bod, 2003) on the entire testing set.
---------------------------------------------------
P04-1013:271	21:185	2 Two History-Based Probability Models As with many previous statistical parsers (Ratnaparkhi, 1999; Collins, 1999; Charniak, 2000), we use a history-based model of parsing.
---------------------------------------------------
N03-1014:272	186:204	Ratnaparkhi (1999) defines a very general set of features for the histories of a shift-reduce parsing model, but the results are not as good as models which use a more linguistically informed set of features for a top-down parsing model (Collins, 1999; Charniak, 2000).
---------------------------------------------------
N03-1014:273	169:204	The concept of lexical head is central to theories of syntax, and has often been used in designing hand-crafted history features (Collins, 1999; Charniak, 2000).
---------------------------------------------------
N03-1014:274	8:204	Many statistical parsers (Ratnaparkhi, 1999; Collins, 1999; Charniak, 2000) are based on a history-based model of parser actions.
---------------------------------------------------
N03-1014:275	140:204	The bottom panel of table 1 lists the results for the two lexicalized models (SSN-Freq a1 200 and SSN-Freq a1 20) and five recent statistical parsers (Ratnaparkhi, 1999; Collins, 1999; Charniak, 2000; Collins, 2000; Bod, 2001).
---------------------------------------------------
N03-1014:276	58:204	The standard way to handle this problem is to hand-craft a finite set of features which provides a sufficient summary of the history (Ratnaparkhi, 1999; Collins, 1999; Charniak, 2000).
---------------------------------------------------
N03-1014:277	12:204	Previous approaches have used a hand-crafted finite set of features to represent the parse history (Ratnaparkhi, 1999; Collins, 1999; Charniak, 2000).
---------------------------------------------------
N03-1014:278	185:204	8 Related Work Most previous work on statistical parsing has used a history-based probability model with a hand-crafted set of features to represent the derivation history (Ratnaparkhi, 1999; Collins, 1999; Charniak, 2000).
---------------------------------------------------
P02-1025:279	42:161	a74 previous results (Chelba and Jelinek, 2000) (Charniak, 2001) (Roark, 2001) show that a grammar-based language model benefits from interpolation with a 3-gram model.
---------------------------------------------------
P02-1025:280	43:161	Strict left-to-right parsing makes it easy to combine with a standard 3-gram at the word level (Chelba and Jelinek, 2000) (Roark, 2001) rather than at sentence level (Charniak, 2001).
---------------------------------------------------
P02-1025:281	70:161	Since the SLM parses sentences bottom-up while the parsers used in (Charniak, 2000), (Charniak, 2001) and (Roark, 2001) are top-down, its not clear how to find a direct correspondence between our schemes of enriching the dependency structure and the ones employed above.
---------------------------------------------------
P02-1025:282	9:161	The statistical parsing community has used various ways of enriching the dependency structure underlying the parametrization of the probabilistic model used for scoring a given parse tree (Charniak, 2000) (Collins, 1999).
---------------------------------------------------
P05-1024:283	139:200	(Charniak, 2000) extends PCFG and achieves similar performance to (Collins, 2000).
---------------------------------------------------
P05-1024:284	150:200	CH00 = (Charniak, 2000), CO00=(Collins, 2000).
---------------------------------------------------
P02-1047:285	24:159	To build such systems, we train a family of Naive Bayes classifiers on a large set of examples that are generated automatically from two corpora: a corpus of 41,147,805 English sentences that have no annotations, and BLIPP, a corpus of 1,796,386 automatically parsed English sentences (Charniak, 2000), which is available from the Linguistic Data Consortium (www.ldc.upenn.edu).
---------------------------------------------------
P02-1047:286	102:159	To each text span in the BLIPP corpus corresponds a parse tree (Charniak, 2000).
---------------------------------------------------
P02-1047:287	61:159	The second, called BLIPP, is a corpus of only 1,796,386 sentences that were parsed automatically by Charniak (2000).
---------------------------------------------------
W07-2218:288	18:279	(Charniak, 2000; Collins, 1999; Nivre et al. , 2004)).
---------------------------------------------------
W07-2218:289	9:279	All the most accurate dependency parsing models are fully discriminative, unlike constituent parsing where all the state of the art methods have a generative component (Charniak and Johnson, 2005; Henderson, 2004; Collins, 2000).
---------------------------------------------------
E09-1049:290	38:164	Meanwhile, the target of the training corpus is parsed with Charniaks parser (Charniak, 2000), and each phrase is annotated with the constituent that spans the target side of the rules.
---------------------------------------------------
E09-1049:291	100:164	The target side (English) of the training corpus was parsed with the Charniaks parser (Charniak, 2000).
---------------------------------------------------
D09-1120:292	38:207	Unlabeled: (see Section 3.2)  BLIPP: 1.8 million sentences of newswire parsed with the Charniak (2000) parser.
---------------------------------------------------
P09-1006:293	102:212	We employed Charniaks maximum entropy inspired parser (Charniak, 2000) to generate N-best (N=200) parses.
---------------------------------------------------
W05-1514:294	178:207	LR LP F-score Ratnaparkhi (1997) 86.3 87.5 86.9 Collins (1999) 88.1 88.3 88.2 Charniak (2000) 89.6 89.5 89.5 Kudo (2005) 89.3 89.6 89.4 Sang (2001) 78.7 82.3 80.5 Deterministic (tagger-POSs) 81.2 86.5 83.8 Deterministic (gold-POSs) 82.6 87.7 85.1 Search (tagger-POSs) 83.2 87.1 85.1 Search (gold-POSs) 84.6 88.5 86.5 Iterative Search (tagger-POSs) 85.0 86.8 85.9 Iterative Search (gold-POSs) 86.2 88.0 87.1 Table 6: Comparison with other work.
---------------------------------------------------
W05-1514:295	13:207	However, there is a very large gap between their performance and that of widely-used practical parsers (Charniak, 2000; Collins, 1999).
---------------------------------------------------
P05-1013:296	130:139	Although the best published results for the Collins parser is 80% UAS (Collins, 1999), this parser reaches 82% when trained on the entire training data set, and an adapted version of Charniaks parser (Charniak, 2000) performs at 84% (Jan Hajic, pers.
---------------------------------------------------
J05-1003:297	65:603	First, several of the best-performing parsers on the WSJ treebank (e.g. , Ratnaparkhi 1997; Charniak 1997, 2000; Collins 1997, 1999; Henderson 2003) are cases of history-based models.
---------------------------------------------------
J05-1003:298	505:603	The model in Charniak (2000) is quite different, however.
---------------------------------------------------
J05-1003:299	504:603	Our features are in many ways similar to those of Charniak (2000).
---------------------------------------------------
J05-1003:300	439:603	The method gives very similar accuracy to the model of Charniak (2000), which also uses a rich set of initial features in addition to Charniaks (1997) original model.
---------------------------------------------------
J05-1003:301	502:603	Related Work 6.1 History-Based Models with Complex Features Charniak (2000) describes a parser which incorporates additional features into a previously developed parser, that of Charniak (1997).
---------------------------------------------------
W03-1005:302	5:166	1 Introduction Many broad-coverage statistical parsers (Charniak, 2000; Collins, 1999; Bod, 2001) are not able to give a full interpretation for sentences such as: (1) It is difficult to guess what she wants to buy.
---------------------------------------------------
W08-2102:303	20:209	For example, the model in (Taskar et al., 2004) is trained on only sentences of 15 words or less; reranking models (Collins, 2000; Charniak and Johnson, 2005) restrict Y(x) to be a small set of parses from a first-pass parser; see section 1.1 for discussion of other related work.
---------------------------------------------------
W08-2102:304	185:209	The Charniak (2000) model is also arguably more complex, again using a carefully constructed generative model.
---------------------------------------------------
W08-2102:305	189:209	Charniak and Johnson (2005), and Huang (2008), describe approaches that make use of nonlocal features in conjunction with the Charniak (2000) model; future work may consider extending our approach to include non-local features.
---------------------------------------------------
W08-2102:306	35:209	In reranking approaches, a first-pass parser is used to enumerate a small set of candidate parses for an input sentence; the reranking model, which is a GLM, is used to select between these parses (e.g., (Ratnaparkhi et al., 1994; Johnson et al., 1999; Collins, 2000; Charniak and Johnson, 2005)).
---------------------------------------------------
W08-2102:307	183:209	Our experiments show an improvement in performance over the results in (Collins, 2000; Charniak, 2000).
---------------------------------------------------
W08-2102:308	31:209	Experiments on the Penn WSJ treebank show that the model recovers constituent structures with higher accuracy than the approaches of (Charniak, 2000; Collins, 2000; Petrov and Klein, 2007), and with a similar level of performance to the reranking parser of (Charniak and Johnson, 2005).
---------------------------------------------------
W08-2102:309	159:209	(2008), Charniak (2000), Collins (2000), Petrov and Klein (2007), Charniak and Johnson (2005), and Huang (2008).
---------------------------------------------------
J09-2003:310	69:340	8 Then, only those English sentences which appeared verbatim in all four language pairs were considered.The resulting English corpus contained 10,000 sentences which were syntactically parsed using Charniaks parser (Charniak 2000).From these we extracted 6,200 token instances of N N (49.62%) and N P N (50.38%) constructions.
---------------------------------------------------
J09-2003:311	70:340	CLUVI (Linguistic Corpus of the University of Vigo) is an open text repository of parallel corpora of contemporary oral and written languages, a resource that besides Galician also contains literary text collections in other Romance languages.Because the collection provides translations into only two of the Romance languages considered here, Spanish and Portuguese, we focused only on the EnglishPortuguese and English Spanish literary parallel texts from the works of Agatha Christie, James Joyce, and H.G. Wells, among others.Using the CLUVI search interfaces we created a sentence-aligned parallel corpus of 4,800 unique EnglishPortugueseSpanish sentences.The English version was syntactically parsed using Charniaks parser (Charniak 2000) after which each N N and N P N instance was manually mapped to the corresponding translations.
---------------------------------------------------
D07-1062:312	47:201	For the test data, we report on results using the gold-standardTreebankdata,andinadditionwealso report results on automatically parsed data using the Charniak parser (Charniak, 2000) as provided by the CoNLL 2005 shared task.
---------------------------------------------------
D07-1062:313	128:201	Tofullyevaluatetheinfluenceofthe LTAG-basedfeatures,wereporttheidentificationresults on both Gold Standard parses and on Charniak parser output (Charniak, 2000)5.
---------------------------------------------------
P06-2048:314	5:249	1 Introduction Much of the current research into probabilistic parsing is founded on probabilistic contextfree grammars (PCFGs) (Collins, 1996; Charniak, 1997; Collins, 1999; Charniak, 2000; Charniak, 2001; Klein and Manning, 2003).
---------------------------------------------------
H05-1099:315	70:185	2.2 Shallow Parser In addition to the trainable n-best context-free parser from Charniak (2000), we needed a trainable shallow parser to apply to the variety of tasks we were interested in investigating.
---------------------------------------------------
H05-1099:316	78:185	The one-best performance of the parser is the same as what was presented in Charniak (2000).
---------------------------------------------------
H05-1099:317	11:185	The output of a contextfree parser, such as that of Collins (1997) or Charniak (2000), can be transformed into a sequence of shallow constituents for comparison with the output of a shallow parser.
---------------------------------------------------
H05-1099:318	63:185	Collins (2000) reported a reranking model that improved his parser output to roughly the same level of accuracy as Charniak (2000), and Charniak and Johnson (2005) report an improvement using reranking over Charniak (2000).
---------------------------------------------------
H05-1099:319	42:185	Perhaps the most widely accepted convention is that of ignoring punctuation for the purposes of assigning constituent span, under the perspective that, fun788 Phrase Evaluation Scenario System Type (a) (b) (c) Modified All 98.37 99.72 99.72 Truth VP 92.14 98.70 98.70 Li and Roth All 94.64 (2001) VP 95.28 Collins (1997) All 92.16 93.42 94.28 VP 88.15 94.31 94.42 Charniak All 93.88 95.15 95.32 (2000) VP 88.92 95.11 95.19 Table 1: F-measure shallow bracketing accuracy under three different evaluation scenarios: (a) baseline, used in Li and Roth (2001), with original chunklink script converting treebank trees and context-free parser output; (b) same as (a), except that empty subject NPs are inserted into every unary SVP production; and (c) same as (b), except that punctuation is ignored for setting constituent span.
---------------------------------------------------
H05-1099:320	68:185	mentation of the Collins parser and the n-best version of the Charniak (2000) parser, documented in Charniak and Johnson (2005), fit the requirements.
---------------------------------------------------
H05-1099:321	151:185	The Charniak and Johnson (2005) system output (denoted C & J in the table) before reranking (denoted one-best) is identical to the Charniak (2000) results that have been reported in the other tables.
---------------------------------------------------
H05-1099:322	133:185	Note that the systems used here are exactly the ones presented for the original Li & Roth task, in SecPunctuation System Leave Ignore Li & Roth (reference tags) 88.47 SPRep avg perceptron Reference tags 91.37 91.86 Brill tags 87.94 88.42 Charniak (2000) 87.94 88.44 Unweighted intersection 88.66 89.16 Weighted intersection 89.22 89.69 Table 4: Shallow bracketing accuracy of several different systems, trained on sections 2-21 of WSJ Treebank and applied to section 4 of the Switchboard Treebank.
---------------------------------------------------
H05-1099:323	97:185	3 Experimental Results 3.1 Comparing Finite-State and Context-Free Parsers The first two rows of Table 3 present a comparison between the SPRep shallow parser and the Charniak (2000) context-free parser detailed in Charniak and Johnson (2005).
---------------------------------------------------
H05-1099:324	62:185	Of the two parsers we evaluated, the Charniak (2000) parser gave the best performance, which is consistent with its higher reported performance on the context-free parsing task versus other context-free parsers.
---------------------------------------------------
H05-1099:325	120:185	Again, the best-scoring candidate using this composite score is selected from among the shallow 791 NP-Chunking CoNLL-2000 Li & Roth task Punctuation Punctuation Punctuation System Leave Ignore Leave Ignore Leave Ignore SPRep averaged perceptron 94.21 94.25 93.54 93.70 95.12 95.27 Charniak (2000) 94.17 94.20 93.77 93.92 95.15 95.32 Unweighted intersection 95.13 95.16 94.52 94.64 95.77 95.92 Weighted intersection 95.57 95.58 95.03 95.16 96.20 96.33 Table 3: F-measure shallow bracketing accuracy on three shallow parsing tasks, for the SPRep perceptron shallow parser, the Charniak (2000) context-free parser, and for systems combining the SPRep and Charniak system outputs.
---------------------------------------------------
H05-1099:326	100:185	For these trials, we used just the one-best output of that model, which is the same as in Charniak (2000).
---------------------------------------------------
H05-1033:327	210:234	Both the baseline and Spade operate on parse trees which were obtained from Charniaks (2000) parser.
---------------------------------------------------
H05-1033:328	16:234	Another important reason is the development of robust syntactic parsers (e.g. , Charniak, 2000) that can be used to provide critical structural and lexical information to the discourse parser.
---------------------------------------------------
D09-1135:329	13:227	Lexicalized PCFGs use the structural features on the lexical head of phrasal node in a tree, and get significant improvements for parsing (Collins, 1997; Charniak, 1997; Collins, 1999; Charniak, 2000).
---------------------------------------------------
D09-1135:330	52:227	(2008) train two lexicalized models (Charniak, 2000; Bikel, 2004) on preprocessed inputs, where content words are substituted with semantic classes from WordNet.
---------------------------------------------------
W05-1518:331	46:283	English version described in Charniak (2000), Czech adaptation 2002  2003, unpublished.
---------------------------------------------------
W05-0211:332	74:173	In (Charniak, 2000), he proposes a generative model based on a Markov-grammar.
---------------------------------------------------
P08-2054:333	7:74	Treebank-specificheuristicshavecommonlybeen used both to alleviate inadequate independence assumptions stipulated by naive PCFGs (Collins, 1999; Charniak, 2000).
---------------------------------------------------
C08-1027:334	105:249	The English side is parsed using a state-of-the-art statistical English parser (Charniak, 2000).
---------------------------------------------------
C00-2135:335	15:148	The progress in parsing technology are noteworthy, and in particular, various statistical dependency models have been proposed(Collins, 1997),, (Ratnaparkhi, 1997), (Charniak, 2000).
---------------------------------------------------
H05-2004:336	21:45	Next, sentences are analyzed by a state-of-the-art syntactic parser (Charniak, 2000) the output of which provides useful information for the main SRL module.
---------------------------------------------------
P06-3014:337	126:139	All these results seem to suggest that adding punctuation in speech transcription is of little help to statistical parsers including at least three state-of-the-art statistical parsers (Collins, 1999; Charniak, 2000; Bikel, 2004).
---------------------------------------------------
P06-3014:338	7:139	1 Introduction Robust statistical syntactic parsers, made possible by new statistical techniques (Collins, 1999; Charniak, 2000; Bikel, 2004) and by the availability of large, hand-annotated training corpora such as WSJ (Marcus et al. , 1993) and Switchboard (Godefrey et al. , 1992), have had a major impact on the field of natural language processing.
---------------------------------------------------
P07-1120:339	9:209	Pipeline systems are ubiquitous in natural language processing, used not only in parsing (Ratnaparkhi, 1999; Charniak, 2000), but also machine translation(OchandNey, 2003)andspeechrecognition (Fiscus, 1997; Goel et al. , 2000), among others.
---------------------------------------------------
P07-1120:340	30:209	The Charniak parsing pipeline has been extensively studied over the past decade, with a number of papers focused on improving early stages of the pipeline (Charniak et al. , 1998; Caraballo and Charniak, 1998; Blaheta and Charniak, 1999; Hall and Johnson, 2004; Charniak et al. , 2006) as well as many focused on optimizing final parse accuracy (Charniak, 2000; Charniak and Johnson, 2005; McClosky et al. , 2006).
---------------------------------------------------
P07-1120:341	33:209	The well-known Charniak (2000) coarse-to-fine parser is a two-stage parsing pipeline, in which the first stage uses a vanilla PCFG to populate a chart of parse constituents.
---------------------------------------------------
N06-1020:342	51:208	3.1 The first-stage 50-best parser The first stage of our parser is the lexicalized probabilistic context-free parser described in (Charniak, 2000) and (Charniak and Johnson, 2005).
---------------------------------------------------
N06-1020:343	8:208	Given sufficient labelled data, there are several supervised techniques of training high-performance parsers (Charniak and Johnson, 2005; Collins, 2000; Henderson, 2004).
---------------------------------------------------
W07-1209:344	67:185	This parser uses a discriminative reranker that selects the most probable parse from the 50-best parses returned by a generative parser based on Charniak (2000).
---------------------------------------------------
W05-1513:345	10:152	Although state-of-the-art statistical parsers (Collins, 1997; Charniak, 2000) are more accurate, the simplicity and efficiency of deterministic parsers make them attractive in a number of situations requiring fast, light-weight parsing, or parsing of large amounts of data.
---------------------------------------------------
W05-1513:346	133:152	Table 1 shows a summary of the results of our experiments with SVMpar and MBLpar, and also results obtained with the Charniak (2000) parser, the Bikel (2003) implementation of the Collins (1997) parser, and the Ratnaparkhi (1997) parser.
---------------------------------------------------
P07-1062:347	14:192	Their models and algorithm  subsequently packaged together into the publicly available SPADE discourse parser1  make use of the output of the Charniak (2000) parser to derive syntactic indicator features for segmentation and discourse parsing.
---------------------------------------------------
P07-1062:348	65:192	Secondly, Charniak and Johnson (2005) showed how reranking of the 50best output of the Charniak (2000) parser gives substantial improvements in parsing accuracy.
---------------------------------------------------
W07-0208:349	26:151	Here, solid lines correspond to surface syntactic structure, produced by Charniaks parser (Charniak, 2000), and dashed lines are an encoding of the Proposition Bank annotation of the semantic roles with respect to the verb stopped.
---------------------------------------------------
W07-0208:350	30:151	Consider the task of recovering non-local dependencies (such as control, WH-extraction, topicalization) in the surface syntactic phrase trees produced by the state-of-the-art parser of (Charniak, 2000).
---------------------------------------------------
N03-1011:351	91:172	Each sentence in this corpus was then parsed using the syntactic parser developed by Charniak (Charniak, 2000).
---------------------------------------------------
H05-1078:352	64:199	Following (Blaheta and Charniak, 2000), we refer to the first class as syntactic function labels, and to the second class as semantic function labels.
---------------------------------------------------
H05-1078:353	47:199	As with many other statistical parsers (Collins, 1999; Charniak, 2000), the model of parsing is history-based.
---------------------------------------------------
H05-1078:354	66:199	Like previous work (Blaheta and Charniak, 2000), we complete the sets of syntactic and semantic labels by labelling constituents that do not bear any function label with a NULL label.3 2.3 Evaluation To evaluate the performance of our function parsing experiments, we will use several measures.
---------------------------------------------------
H05-1078:355	22:199	tion labels are context-dependent and encode a shallow level of phrasal and lexical semantics, as observed first in (Blaheta and Charniak, 2000).1 To a large extent, they overlap with semantic role labels as defined in PropBank (Palmer et al. , 2005).
---------------------------------------------------
H05-1078:356	17:199	Statistical parsers trained on the Penn Treebank (PTB) (Marcus et al. , 1993) produce trees annotated with bare phrase structure labels (Collins, 1999; Charniak, 2000).
---------------------------------------------------
H05-1078:357	30:199	While the function of a constituent and its structural position are often correlated, they some1(Blaheta and Charniak, 2000) talk of function tags.
---------------------------------------------------
H05-1078:358	181:199	First, they parse the Penn Treebank using a state-of-the-art parser (Charniak, 2000).
---------------------------------------------------
H05-1078:359	179:199	In work that predates the availability of Framenet and Propbank, (Blaheta and Charniak, 2000) define the task of function labelling for the first time and highlight its relevance for NLP.
---------------------------------------------------
H05-1078:360	171:199	We will therefore discuss separately those pieces of work that have made limited use of function labels for parsing (Klein and Manning, 2003), and those that have concentrated on recovering function labels as a separate task (Blaheta and Charniak, 2000; Blaheta, 2004).
---------------------------------------------------
H05-1078:361	79:199	Following (Blaheta and Charniak, 2000), incorrectly parsed constituents will be ignored (roughly 11% of the total) in the evaluation of the precision and recall of the function labels, but not in the evaluation of the parser.
---------------------------------------------------
W01-0714:362	7:226	Because of these kinds of results, the vast majority of statistical parsing work has focused on parsing as a supervised learning problem (Collins, 1997; Charniak, 2000).
---------------------------------------------------
C08-1094:363	8:177	The Charniak (2000) parser uses a simple PCFG to prune the chart for a richer model; and Charniak and Johnson (2005) added a discriminatively trained reranker to the end of that pipeline.
---------------------------------------------------
C08-1094:364	97:177	5 Constraining the Charniak parser 5.1 Parser overview and constraint methods The Charniak (2000) parser is a multi-stage, agenda-driven, edge-based parser, that can be constrained by precluding edges from being placed on the agenda.
---------------------------------------------------
W04-2407:365	129:153	1999; Charniak, 2000).
---------------------------------------------------
W04-2407:366	86:153	Thus, the Penn Treebank of American English (Marcus et al. , 1993) has been used to train and evaluate the best available parsers of unrestricted English text (Collins, 1999; Charniak, 2000).
---------------------------------------------------
W04-2002:367	49:60	This finding is in line with Charniaks own analysis, which shows that the high performance of his model is due to the fact that it combines a thirdorder Markov grammar with sophisticated phrasal and lexical features (Charniak, 2000).
---------------------------------------------------
W04-2002:368	39:60	These models include a standard unlexicalized PCFG parser, a head-lexicalized parser (Collins, 1997), and a maximum-entropy inspired parser (Charniak, 2000).
---------------------------------------------------
P02-1018:369	4:149	Evaluating the algorithm on the output of Charniaks parser (Charniak, 2000) and the Penn treebank (Marcus et al. , 1993) shows that the patternmatching algorithm does surprisingly well on the most frequently occuring types of empty nodes given its simplicity.
---------------------------------------------------
P02-1018:370	8:149	Broad coverage syntactic parsers with good performance have recently become available (Charniak, 2000; Collins, 2000), but these typically produce as output a parse tree that only encodes local syntactic information, i.e., a tree that does not include any empty nodes.
---------------------------------------------------
P02-1018:371	115:149	Then as is standard, the precision P, recall R and f-score f are calculated as follows: P = jG \ TjjTj R = jG \ TjjGj f = 2 P RP + R Table 3 provides these measures for two different test corpora: (i) a version of section 23 of the Penn Treebank from which empty nodes, indices and unary branching chains consisting of nodes of the same category were removed, and (ii) the trees produced by Charniaks parser on the strings of section 23 (Charniak, 2000).
---------------------------------------------------
P02-1018:372	129:149	The parser output was produced by Charniaks parser (Charniak, 2000).
---------------------------------------------------
J08-1002:373	329:587	Following previous studies on parsing with PCFG-based models (Collins 1997; Charniak 2000), accuracy is measured for sentences of less than 40 words and for those with less than 100 words.
---------------------------------------------------
J08-1002:374	556:587	We should also investigate compromise solutions such as dynamic CRFs (McCallum, Rohanimanesh, and Sutton 2003; Sutton, Rohanimanesh, and McCallum 2004) and reranking techniques (Collins 2000; Charniak and Johnson 2005).
---------------------------------------------------
J08-1002:375	367:587	For reference, our results are competitive with the best corresponding results reported in CCG parsing (LP/LR = 86.6/86.3) (Clark and Curran 2004b), although our results cannot be compared directly with other grammar formalisms because each formalismrepresentspredicateargumentdependenciesdifferently.Incontrastwiththe results of CCG and PCFG (Collins 1997, 1999, 2003; Charniak 2000), the recall is clearly lower than precision.
---------------------------------------------------
P06-1117:376	176:216	The sentences were processed using Charniaks parser (Charniak, 2000) to generate parse trees automatically.
---------------------------------------------------
J05-1004:377	7:501	Introduction Robust syntactic parsers, made possible by new statistical techniques (Ratnaparkhi 1997; Collins 1999, 2000; Bangalore and Joshi 1999; Charniak 2000) and by the availability of large, hand-annotated training corpora (Marcus, Santorini, and Marcinkiewicz 1993; Abeille 2003), have had a major impact on the field of natural language processing in recent years.
---------------------------------------------------
W04-0824:378	39:92	Training and test data, and the Wordnet glosses, were parsed with Charniaks parser (Charniak, 2000).
---------------------------------------------------
P07-2057:379	10:124	Generally, the syntactic structure of a sentence is represented as a tree, and parsing is carried out by maximizing the likelihood of the tree (Charniak, 2000; Uchimoto et al. , 1999).
---------------------------------------------------
W06-0901:380	72:160	These sentences are parsed using the August 2005 release of the Charniak parser (Charniak, 2000)4.
---------------------------------------------------
D07-1027:381	10:285	However, with few exceptions (Model 3 of Collins, 1999; Schmid, 2006), output trees produced by state-of-the-art broad coverage statistical parsers (Charniak, 2000; Bikel, 2004) are only surface context-free phrase structure trees (CFG-trees) without empty categories and coindexation to represent displaced constituents.
---------------------------------------------------
W03-0402:382	208:239	The performance of our system matches the results of (Charniak, 2000), but is a little lower than the results of the Boosting system in (Collins, 2000), except that the percentage of sentences with no crossing brackets is 1% higher than that of (Collins, 2000).
---------------------------------------------------
W03-0402:383	196:239	CH00 = (Charniak, 2000).
---------------------------------------------------
N04-1020:384	58:249	3 Parameter Estimation 3.1 Data Extraction Subordinate clauses (and their main clause counterparts) were extracted from the BLLIP corpus (30 M words), a Treebank-style, machine-parsed version of the Wall Street Journal (WSJ, years 198789) which was produced using Charniaks (2000) parser.
---------------------------------------------------
W09-0103:385	65:176	My guess is that the features used in e.g., the Collins (2003) or Charniak (2000) parsers are probably close to optimal for English Penn Treebank parsing (Marcus et al., 1993), but that other features might improve parsing of other languages or even other English genres.
---------------------------------------------------
P05-1023:386	139:176	For comparison to previous results, table 2 lists the results on the testing set for our best model (TOP-Efficient-Freq20) and several other statistical parsers (Collins, 1999; Collins and Duffy, 2002; Collins and Roark, 2004; Henderson, 2003; Charniak, 2000; Collins, 2000; Shen and Joshi, 2004; Shen et al. , 2003; Henderson, 2004; Bod, 2003).
---------------------------------------------------
P05-1023:387	65:176	3.1 A History-Based Probability Model As with many other statistical parsers (Ratnaparkhi, 1999; Collins, 1999; Charniak, 2000), Henderson (2003) uses a history-based model of parsing.
---------------------------------------------------
N03-1024:388	77:235	Given a sentence group, we pass each of the 11 sentences to Charniaks (2000) parser to get 11 parse trees.
---------------------------------------------------
W01-0712:389	204:210	The preliminary results are encouraging though not as good as advanced statistical parsers like those of Charniak (2000) and Collins (2000).
---------------------------------------------------
C04-1180:390	5:129	1 Introduction The levels of accuracy and robustness recently achieved by statistical parsers (e.g. Collins (1999), Charniak (2000)) have led to their use in a number of NLP applications, such as question-answering (Pasca and Harabagiu, 2001), machine translation (Charniak et al. , 2003), sentence simplification (Carroll et al. , 1999), and a linguists search engine (Resnik and Elkiss, 2003).
---------------------------------------------------
P04-1087:391	91:214	These sentences were then parsed using a statistical parser (Charniak, 2000).
---------------------------------------------------
P02-1055:392	13:174	Charniak (2000) notes that having his generative parser generate the POS of a constituents head before the head itself increases performance by 2 points.
---------------------------------------------------
W05-0638:393	24:90	In CoNLL-2005, full parsing trees are provided by two full parsers: the Collins parser (Collins, 1999) and the Charniak parser (Charniak, 2000).
---------------------------------------------------
W09-1701:394	105:235	3.1.3 Parse and Match The results we want to achieve in this step should describe a relationship: nounA is [in|on] nounB We use Charniaks parser (Charniak, 2000) on both the web query and the results returned from the web in order to ensure accuracy.
---------------------------------------------------
E06-1038:395	114:236	(2005b) parser and phrase structure tree from the Charniak (2000) parser.
---------------------------------------------------
E06-1038:396	8:236	We focus on the particular instantiation of sentence compression when the goal is to produce the compressed version solely by removing words or phrases from the original, which is the most common setting in the literature (Knight and Marcu, 2000; Riezler et al. , 2003; Turner and Charniak, 2005).
---------------------------------------------------
E06-1038:397	110:236	To do this we parse every sentence twice, once with a dependency parser (McDonald et al. , 2005b) and once with a phrase-structure parser (Charniak, 2000).
---------------------------------------------------
D09-1161:398	9:255	In general, they can be divided into two major categories, namely lexicalized models (Collins 1997, 1999; Charniak 1997, 2000) and un-lexicalized models (Klein and Manning 2003; Matsuzaki et al. 2005; Petrov et al. 2006; Petrov and Klein 2007).
---------------------------------------------------
D09-1161:399	182:255	P means precision, R means recall and F is the F1-measure (all is in % percentage metrics); Charniak represents the parser of (Charniak 2000), Berkeley represents the parser of (Petrov and Klein 2007), Comb. represents the combination of the two parsers.
---------------------------------------------------
D09-1161:400	80:255	Charniak (2000)s model introduces pre-terminal as additional features.
---------------------------------------------------
D09-1161:401	246:255	System  F1-Measure I Charniak (2000) 80.85 Petrov and Klein (2007) 83.13 B Burkett and Klein (2008) 1  84.24 C Our method 85.45  Table 11.
---------------------------------------------------
D09-1161:402	244:255	Performance with Charniak parser enhanced by re-ranking plus self-training 5.7 Comparison with Other State-of-the-art Results Table 11 and table 12 compare our method with the other state-of-the-art methods; we use I, B, R, S and C to denote individual model (Charniak 2000; Collins 2000; Bod 2003; Petrov and Klein 2007), bilingual-constrained model (Burkett and Klein 2008) 1 , re-ranking model (Charniak and Johnson 2005, Huang 2008), self-training model (David McClosky 2006) and combination model (Sagae and Lavie 2006) respectively.
---------------------------------------------------
D09-1161:403	165:255	We use Charniaks parser (Charniak 2000) and Berkeleys parser (Petrov and Klein 2007) as the two individual parsers, where Charniaks parser represents the best performance of the lexicalized model and the Berkeleys parser represents the best performance of the un-lexicalized model.
---------------------------------------------------
D09-1161:404	248:255	System  F1-Measure I Petrov and Klein (2007) 89.5 Charniak (2000) 89.7 Bod (2003) 90.7 R Collins (2000) 89.7 Charniak and Johnson (2005) 91.4 Huang (2008) 91.7 S David McClosky (2006) 92.1 C Sagae and Lavie (2006) 92.1 Our method 92.6    Table 12.
---------------------------------------------------
D09-1161:405	14:255	In addition, parsing re-ranking (Collins 2000; Riezler et al. 2002; Charniak and Johnson 2005; Huang 2008) has also been shown to be another effective technique to improve parsing performance.
---------------------------------------------------
D09-1161:406	179:255	The experiment is set up as follows: 1) for each sentence in the dev and test sets, we generate 50-best from Charniaks parser (Charniak 2000) and Berkeleys parser (Petrov and Klein 2007), respectively; 2) the two 50-best trees are merged together and duplication was removed; 3) we tune the parameters on the dev set and test on the test set.
---------------------------------------------------
P05-1038:407	159:222	(1999) for Czech, where the bigram model upped dependency accuracy by about 0.9%, as well as for English where Charniak (2000) reports an increase in F-score of approximately 0.3%.
---------------------------------------------------
P06-2038:408	102:105	299 4 Future Work While the modi cation given in section 2.2 is speci c to CYK parsing, we believe that placing restrictions based on the output of a chunk parser is general enough to be applied to any generative, statistical parser, such as the Charniak parser (2000), or a Lexical Tree Adjoining Grammar based parser (Sarkar, 2000).
---------------------------------------------------
P05-1012:409	5:209	The best phrase-structure parsing models represent generatively the joint probability P(x,y) of sentence x having the structure y (Collins, 1999; Charniak, 2000).
---------------------------------------------------
P05-1012:410	174:209	3.1 Lexicalized Phrase Structure Parsers It is well known that dependency trees extracted from lexicalized phrase structure parsers (Collins, 1999; Charniak, 2000) typically are more accurate than those produced by pure dependency parsers (Yamada and Matsumoto, 2003).
---------------------------------------------------
P06-1048:411	17:244	Many algorithms exploit parallel corpora (Jing 2000; Knight and Marcu 2002; Riezler et al. 2003; Nguyen et al. 2004a; Turner and Charniak 2005; McDonald 2006) to learn the correspondences between long and short sentences in a supervised manner, typically using a rich feature space induced from parse trees.
---------------------------------------------------
P06-1048:412	157:244	RASP failed on 17 sentences from the Broadcast news corpus and 33 from the Ziff-Davis corpus; Charniaks (2000) parser successfully parsed the Broadcast News corpus but failed on three sentences from the Ziff-Davis corpus.
---------------------------------------------------
P06-1048:413	145:244	The training data for both models was parsed using Charniaks (2000) parser.
---------------------------------------------------
P06-1048:414	102:244	Ziff-Davis Corpus Most previous work (Jing 2000; Knight and Marcu 2002; Riezler et al. 2003; Nguyen et al. 2004a; Turner and Charniak 2005; McDonald 2006) has relied on automatically constructed parallel corpora for training and evaluation purposes.
---------------------------------------------------
W03-1022:415	120:201	We parsed the WordNet definitions and example sentences with the same syntactic parser used for Bllip (Charniak, 2000).
---------------------------------------------------
P07-1032:416	8:189	1 Introduction Parsers have been developed for a variety of grammar formalisms, for example HPSG (Toutanova et al. , 2002; Malouf and van Noord, 2004), LFG (Kaplan et al. , 2004; Cahill et al. , 2004), TAG (Sarkar and Joshi, 2003), CCG (Hockenmaier and Steedman, 2002; Clark and Curran, 2004b), and variants of phrase-structure grammar (Briscoe et al. , 2006), including the phrase-structure grammar implicit in the Penn Treebank (Collins, 2003; Charniak, 2000).
---------------------------------------------------
P07-1032:417	38:189	Preiss (2003) compares the parsers of Collins (2003) and Charniak (2000), the GR finder of Buchholz et al.
---------------------------------------------------
D09-1087:418	6:227	1 Introduction There is an extensive research literature on building high quality parsers for English (Collins, 1999; Charniak, 2000; Charniak and Johnson, 2005; Petrov et al., 2006), however, models for parsing other languages are less well developed.
---------------------------------------------------
W02-1009:419	30:288	Most researchers have used an n-gram model (Eisner, 1996; Charniak, 2000) or more general Markov model (Alshawi, 1996) to model the sequence of nonterminals in the RHS.
---------------------------------------------------
C08-1069:420	114:172	We compared the Enju parser with four CFG parsers: Stanfords lexicalized parser (Klein and Manning, 2003), Collins parser (Collins, 1999), Charniaks parser (Charniak, 2000), and Charniak and Johnsons reranking parser (Charniak and Johnson, 2005).
---------------------------------------------------
C04-1204:421	6:135	However, their accuracy was still below the state-of-theart PCFG parsers (Collins, 1999; Charniak, 2000) in terms of the PARSEVAL score.
---------------------------------------------------
C04-1204:422	33:135	However, the accuracy of those parsers was still below PCFG parsers (Collins, 1999; Charniak, 2000) in terms of the PARSEVAL score, i.e., labeled bracketing accuracy of CFG-style parse trees.
---------------------------------------------------
N04-1011:423	19:140	Even though this seems linguistically highly unnatural (e.g. , punctuation might indicate suprasegmental prosodic properties), statistical parsers generally perform signi cantly better when their training and test data contains punctuation represented in this way than if the punctuation is stripped out of the training and test data (Charniak, 2000; Engel et al. , 2002; Johnson, 1998).
---------------------------------------------------
P05-1011:424	114:172	Different from the results of CCG and PCFG (Collins, 1999; Charniak, 2000), the recall was clearly lower than precision.
---------------------------------------------------
C08-1144:425	57:207	Each target sentence in the training corpus is parsed with a stochastic parserwe use Charniak (2000))to produce constituent labels for target spans.
---------------------------------------------------
W09-0809:426	146:263	The English side is parsed using a state-of-the-art statistical English parser (Charniak, 2000).
---------------------------------------------------
W05-0639:427	22:86	In order to attack this problem, in addition to Charniaks parser (Charniak, 2000), our system combine two parser which are trained on both syntactic constituent information and semantic argument information.
---------------------------------------------------
W05-0639:428	19:86	2.1 Parsing Previous SRL systems usually use a pure syntactic parser, such as (Charniak, 2000; Collins, 1999), to retrieve possible constituents.
---------------------------------------------------
N06-1040:429	179:188	niak parser (Charniak, 2000; Charniak and Johnson, 2005) uses a Markov order-3 baseline PCFG in the initial pass, with a best-first algorithm that is run past the first parse to populate the chart for use by the richer model.
---------------------------------------------------
N06-1040:430	31:188	Probability estimates of the RHS given the LHS are often smoothed by making a Markov assumption regarding the conditional independence of a category on those more than k categories away (Collins, 1997; Charniak, 2000): P(X  Y1Yn)= P(Y1|X) nY i=2 P(Yi|X,Y1 Yi1)  P(Y1|X) nY i=2 P(Yi|X,Yik Yi1).
---------------------------------------------------
N06-1040:431	51:188	One common strategy in statistical parsing is what can be termed an approximate coarse-to-fine approach: a simple PCFG is used to prune the search space to which richer and more complex models are applied subsequently (Charniak, 2000; Charniak and Johnson, 2005).
---------------------------------------------------
P05-1022:432	2:180	c2005 Association for Computational Linguistics Coarse-to-fine n-best parsing and MaxEnt discriminative reranking Eugene Charniak and Mark Johnson Brown Laboratory for Linguistic Information Processing (BLLIP) Brown University Providence, RI 02912 {mj|ec}@cs.brown.edu Abstract Discriminative reranking is one method for constructing high-performance statistical parsers (Collins, 2000).
---------------------------------------------------
P05-1022:433	104:180	Finally, we note that 50-best parsing is only a fac1Charniak in (Charniak, 2000) cites an accuracy of 89.5%.
---------------------------------------------------
P05-1022:434	8:180	The 50-best parser is a probabilistic parser that on its own produces high quality parses; the maximum probability parse trees (according to the parsers model) have an f-score of 0.897 on section 23 of the Penn Treebank (Charniak, 2000), which is still state-of-the-art.
---------------------------------------------------
P05-1022:435	56:180	Things become worse still in a parser like the one described in Charniak (2000) because it conditions on (and hence splits the dynamic programming states according to) features of the grandparent node in addition to the parent, thus multiplying the number of possible dynamic programming states even more.
---------------------------------------------------
P05-1022:436	4:180	This paper describes a simple yet novel method for constructing sets of 50-best parses based on a coarse-to-fine generative parser (Charniak, 2000).
---------------------------------------------------
N03-1027:437	63:172	The PCFG is a Markov grammar (Collins, 1997; Charniak, 2000), i.e. the production probabilities are estimated by decomposing the joint probability of the categories on the right-hand side into a product of conditionals via the chain rule, and making a Markov assumption.
---------------------------------------------------
N03-1027:438	115:172	Model interpolation in this case perSystem Training Heldout LR LP MAP Brown;T Brown;H 76.0 75.4 MAP Brown;T WSJ;24 76.9 77.1 Gildea WSJ;2-21 86.1 86.6 MAP WSJ;2-21 WSJ;24 86.9 87.1 Charniak (1997) WSJ;2-21 WSJ;24 86.7 86.6 Ratnaparkhi (1999) WSJ;2-21 86.3 87.5 Collins (1999) WSJ;2-21 88.1 88.3 Charniak (2000) WSJ;2-21 WSJ;24 89.6 89.5 Collins (2000) WSJ;2-21 89.6 89.9 Table 4: Parser performance on WSJ;23, baselines.
---------------------------------------------------
W05-1511:439	77:178	They include grammar compilation (Tomita, 1986; Nederhof, 2000), the Viterbi algorithm, controlling search strategies without FOM such as left-corner parsing (Rosenkrantz and Lewis II, 1970) or headcorner parsing (Kay, 1989; van Noord, 1997), and with FOM such as the beam search, the best-first search or A* search (Chitrao and Grishman, 1990; Caraballo and Charniak, 1998; Collins, 1999; Ratnaparkhi, 1999; Charniak, 2000; Roark, 2001; Klein and Manning, 2003).
---------------------------------------------------
W06-2937:440	20:149	2 System Description 241 Over the past decades, many state-of-the-art parsing algorithm were proposed, such as head-word lexicalized PCFG (Collins, 1998), Maximum Entropy (Charniak, 2000), Maximum/Minimum spanning tree (MST) (McDonald et al. , 2005), Bottom-up deterministic parsing (Yamada and Matsumoto, 2003), and Constant-time deterministic parsing (Nivre, 2003).
---------------------------------------------------
P04-1006:441	46:172	2.1 nbest list reranking Much effort has been put forth in developing efficient probabilistic models for parsing strings (Caraballo and Charniak, 1998; Goldwater et al. , 1998; Blaheta and Charniak, 1999; Charniak, 2000; Charniak, 2001); an obvious solution to parsing wordlattices is to use nbest list reranking.
---------------------------------------------------
P04-1006:442	136:172	on the BLLIP99 corpus (Charniak et al. , 1999); a corpus of 30million words automatically parsed using the Charniak parser (Charniak, 2000).
---------------------------------------------------
P04-1006:443	59:172	In Figure 4 we present the general overview of a multi-stage parsing technique (Goodman, 1997; Charniak, 2000; Charniak, 2001).
---------------------------------------------------
N07-1072:444	59:189	3.3 Preprocessing for Parsing We first used the Charniak parser (2000) to parse the original skill statements.
---------------------------------------------------
W09-1106:445	100:202	5 Experiments We tested our model on a Semantic Role Labeling (SRL) benchmark, using PropBank annotations (Palmer et al., 2005) and automatic Charniak parse trees (Charniak, 2000) as provided for the CoNLL 2005 evaluation campaign (Carreras and M`arquez, 2005).
---------------------------------------------------
P05-1039:446	85:187	If the tags are necessary for semantic interpretation, presumably they could be re-inserted using a strategy such as that of Blaheta and Charniak (2000) The last transformation therefore removes the GF of S nodes.
---------------------------------------------------
P05-1039:447	145:187	This is worrying: at times in the literature, details of search or smoothing are left out (e.g. Charniak (2000)).
---------------------------------------------------
P05-1039:448	46:187	The first is a Markov context-free rule (Magerman, 1995; Charniak, 2000).
---------------------------------------------------
W00-1308:449	174:186	This presumably corresponds with Charniak's (2000: 136) observation that Section 23 of the Penn Treebank is easier than some others.
---------------------------------------------------
W05-0620:450	138:334	Test WSJ Test Brown P(%) R(%) F1 P(%) R(%) F1 P(%) R(%) F1 UPC Chunker 94.66 93.17 93.91 95.26 94.52 94.89 92.64 90.85 91.73 UPC Clauser 90.38 84.73 87.46 90.93 85.94 88.36 84.21 74.32 78.95 Collins (1999) 85.02 83.55 84.28 85.63 85.20 85.41 82.68 81.33 82.00 Charniak (2000) 87.60 87.38 87.49 88.20 88.30 88.25 80.54 81.15 80.84 Table 3: Results of the syntactic parsers on the development, and WSJ and Brown test sets.
---------------------------------------------------
W05-0620:451	111:334	 Full parser of Charniak (2000).
---------------------------------------------------
W05-0620:452	123:334	tWSJ tBrown UPC PoS-tagger 97.13 97.36 94.73 Charniak (2000) 92.01 92.29 87.89 Table 2: Accuracy (%) of PoS taggers.
---------------------------------------------------
W08-1112:453	43:133	3.1 A History-Based Model The history-based (HB) approach which incorporates more context information has worked well in parsing (Collins, 1997; Charniak, 2000).
---------------------------------------------------
W08-1112:454	41:133	Methodologies such as lexicalisation (Collins, 1997; Charniak, 2000) and tree transformations (Johnson, 1998), weaken the independence assumptions and have been applied successfully to parsing and shown significant improvements over simple PCFGs.
---------------------------------------------------
D07-1028:455	13:188	This history-based approach has worked well in parsing (Collins, 1999; Charniak, 2000) and we show that it also improves PCFG-based generation.
---------------------------------------------------
D07-1028:456	97:188	In this way, the generation model resembles history-based models for parsing (Black et al. , 1992; Collins, 1999; Charniak, 2000).
---------------------------------------------------
E09-1017:457	12:229	Considerations of sentence fluency are also key in sentence simplification (Siddharthan, 2003), sentence compression (Jing, 2000; Knight and Marcu, 2002; Clarke and Lapata, 2006; McDonald, 2006; Turner and Charniak, 2005; Galley and McKeown, 2007), text re-generation for summarization (Daume III and Marcu, 2004; Barzilay and McKeown, 2005; Wan et al., 2005) and headline generation (Banko et al., 2000; Zajic et al., 2007; Soricut and Marcu, 2007).
---------------------------------------------------
E09-1017:458	55:229	The sentences were parsed with Charniaks parser (Charniak, 2000) in order to calculate these features.
---------------------------------------------------
P07-1103:459	123:235	Each of these variants was then submitted to a parser trained on written text (Charniak, 2000).
---------------------------------------------------
N09-1073:460	49:180	648 3 Constraining Exact-Inference CYK Despite referring to the CYK algorithm in the proof, in Roark and Hollingshead (2008) we demonstrated our approach by constraining the Charniak parser (Charniak, 2000), and achieved an improvement in the accuracy/efficiency tradeoff curve.
---------------------------------------------------
N09-1073:461	15:180	We demonstrated the application of such constraints to the well-known Charniak parsing pipeline (Charniak,2000),whichresultedinnoaccuracylosswhen the constraints were applied.
---------------------------------------------------
W03-2008:462	9:174	Broad coverage syntactic parsers with good performance have recently become available (Charniak, 2000; Collins, 2000), but they are not trained for patents.
---------------------------------------------------
C04-1010:463	12:293	Moreover, the deterministic dependency parser of Yamada and Matsumoto (2003), when trained on the Penn Treebank, gives a dependency accuracy that is almost as good as that of Collins (1997) and Charniak (2000).
---------------------------------------------------
C04-1010:464	93:293	This permits us to make exact comparisons with the parser of Yamada and Matsumoto (2003), but also the parsers of Collins (1997) and Charniak (2000), which are evaluated on the same data set in Yamada and Matsumoto (2003).
---------------------------------------------------
C04-1010:465	90:293	used for training and section 23 for testing (Collins, 1999; Charniak, 2000).
---------------------------------------------------
C04-1010:466	11:293	On the other hand, the best available parsers trained on the Penn Treebank, those of Collins (1997) and Charniak (2000), use statistical models for disambiguation that make crucial use of dependency relations.
---------------------------------------------------
C04-1010:467	120:293	Table 2 shows the dependency accuracy, root accuracy and complete match scores for our best parser (Model 2 with label set B) in comparison with Collins (1997) (Model 3), Charniak (2000), and Yamada and Matsumoto (2003).5 It is clear that, with respect to unlabeled accuracy, our parser does not quite reach state-of-the-art performance, even if we limit the competition to deterministic methods such as that of Yamada and Matsumoto (2003).
---------------------------------------------------
C04-1010:468	134:293	In another study, Blaheta and Charniak (2000) report an F-measure of 98.9% for the assignment of Penn Treebank grammatical role labels (our G set) to phrases that were correctly parsed by the parser described in Charniak (2000).
---------------------------------------------------
P02-1034:469	141:185	(Charniak 2000) describes a different method which achieves very similar performance to (Collins 2000).
---------------------------------------------------
W07-1220:470	79:193	The manually compiled grammars in our experiment are also intrinsically different to grammars automatically induced from treebanks (e.g. that used in the Charniak parser (Charniak, 2000) or the various CCG parsers (Hockenmaier, 2006)).
---------------------------------------------------
H05-1035:471	20:147	The feature set contains complex information extracted automatically from candidate syntax trees generated by parsing (Charniak, 2000), trees that will be improved by more accurate PP-attachment decisions.
---------------------------------------------------
D08-1050:472	79:252	477 Lease and Charniak (2005) obtained an improvement in the accuracy of the Charniak (2000) parser, as well as POS tagging accuracy, when applied to the biomedical domain, by training a new POS tagger model with a combination of newspaper and biomedical data.
---------------------------------------------------
W06-1636:473	11:210	Instead researchers condition parsing decisions on many other features, such as parent phrase-marker, and, famously, the lexical-head of the phrase (Magerman, 1995; Collins, 1996; Collins, 1997; Johnson, 1998; Charniak, 2000; Henderson, 2003; Klein and Manning, 2003; Matsuzaki et al. , 2005) (and others).
---------------------------------------------------
J01-2004:474	333:462	The differences between a k-best and a beam-search parser (not to mention the use of dynamic programming) make a running time difference unsur17 Our score of 85.8 average labeled precision and recall for sentences less than or equal to 100 on Section 23 compares to: 86.7 in Charniak (1997), 86.9 in Ratnaparkhi (1997), 88.2 in Collins (1999), 89.6 in Charniak (2000), and 89.75 in Collins (2000).
---------------------------------------------------
J01-2004:475	144:462	The parsers with the highest published broad-coverage parsing accuracy, which include Charniak (1997, 2000), Collins (1997, 1999), and Ratnaparkhi (1997), all utilize simple and straightforward statistically based search heuristics, pruning the search-space quite dramatically.
---------------------------------------------------
N03-1030:476	181:187	Another interesting nding is that the performance of current state-of-the-art syntactic parsers (Charniak, 2000) is not a bottleneck for coming up with a good solution to the sentence-level discourse parsing problem.
---------------------------------------------------
N03-1030:477	133:187	Recall Precision F-score a174a113a66a4a112a65a147 28.2 37.1 32.0 a174a74a145a32a112a65a147 25.4 64.9 36.5 a112a113a18a20a11a29a112a170a147 77.1 83.3 80.1 a147a137a3a22a5a117a112a65a147a49a21a95a15a177a183a173a24 82.7 83.5 83.1 a147a62a3a6a5a117a112a170a147a49a21a95a15a44a184a144a24 85.4 84.1 84.7 a149a88a112a170a147 98.2 98.5 98.3 Table 1: Discourse segmenter evaluation Table 1 shows the results obtained by the algorithm described in this paper (a147a62a3a6a5a117a112a170a147a49a21a95a15a104a183a185a24 ) using syntactic trees produced by Charniaks parser (2000), in comparison with the results obtained by the algorithm described in (Marcu, 2000) (a112a113a18a20a11a29a112a170a147 ), and baseline algorithms a174a113a66a27a112a170a147 and a174a74a145a33a112a170a147, on the same test set.
---------------------------------------------------
N03-1030:478	59:187	Given a sentence a2a44a43a46a45a39a47a12a45a49a48a51a50a52a50a4a50a10a45a49a53a54a50a4a50a52a50a55a45a49a56, we rst nd the syntactic parse tree a7 of a2 . We used in our experiments both syntactic parse trees obtained using Charniaks parser (2000) and syntactic parse trees from the PennTree bank.
---------------------------------------------------
N03-1030:479	5:187	1 Introduction By exploiting information encoded in human-produced syntactic trees (Marcus et al. , 1993), research on probabilistic models of syntax has driven the performance of syntactic parsers to about 90% accuracy (Charniak, 2000; Collins, 2000).
---------------------------------------------------
N03-1030:480	122:187	For this evaluation, we re-trained Charniaks parser (2000) such that the test sentences from the discourse corpus were not seen by the syntactic parser during training.
---------------------------------------------------
N03-1030:481	146:187	The discourse parsing model uses syntactic trees produced by Charniaks parser (2000) and discourse segments produced by the algorithm described in Section 3.
---------------------------------------------------
I05-6006:482	144:165	Charniak (2000) uses WSJ as both training and testing data and it is reasonable to expect a fairly good overlap in terms of lexical co-occurrences and linguistic structures and hence good performance scores.
---------------------------------------------------
I05-6006:483	141:165	Charniak (2000) reports a maximum entropy inspired parser that scored 90.1% average precision/recall when trained and tested with sentences from the Wall Street Journal corpus (WSJ).
---------------------------------------------------
I05-2036:484	13:116	All documents are parsed using Eugene Charniaks maximum entropy inspired parser (Charniak, 2000).
---------------------------------------------------
W06-3601:485	63:298	In fact, the recursive transfer step can be done by a a linear-time algorithm (see Section 5), and the parsing step is also fast with the modern Treebank parsers, for instance (Collins, 1999; Charniak, 2000).
---------------------------------------------------
P09-2020:486	70:93	Finally, two parsers are compared for their effect on segmentation quality: Charniak (Charniak, 2000) and Stanford (Klein and Manning, 2003).
---------------------------------------------------
P06-1021:487	16:158	In this separate-processing approach, reparanda are located through a variety of acoustic, lexical or string-based techniques, then excised before submission to a parser (Stolcke and Shriberg, 1996; Heeman and Allen, 1999; Spilker et al. , 2000; Johnson and Charniak, 2004).
---------------------------------------------------
P06-1021:488	99:158	The third experiment measures the benefit from syntactic indicators alone in Charniaks lexicalized parser (Charniak, 2000).
---------------------------------------------------
