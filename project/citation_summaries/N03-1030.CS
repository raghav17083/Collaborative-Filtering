P09-1075:1	102:186	3.4 Dominance Sets A promising concept introduced by Soricut and Marcu (2003) in their sentence-level parser is the identification of dominance sets in the syntax parse trees associated to each input sentence.
---------------------------------------------------
P09-1075:2	45:186	(2001) and previously used by Soricut and Marcu (2003).
---------------------------------------------------
P09-1075:3	24:186	Marcu and Soricut focussed on sentence-level parsing and developed two probabilistic models that use syntactic and lexical information (Soricut and Marcu, 2003).
---------------------------------------------------
P09-1075:4	89:186	3.1 Textual Organization As evidenced by a number of discourse-parsing efforts focusing on intra-sentential parsing (Marcu, 2000; Soricut and Marcu, 2003), there is a strong correlation between different organizational levels of textual units and sub-trees of the RST tree both at the sentence-level and the paragraph level.
---------------------------------------------------
P09-1075:5	110:186	4), Soricut and Marcu (2003) note that in order to achieve good results on relation labeling, [Shoneys Inc. said]1A [it will report a write-off of $2.5 million, or seven cents a share, for its fourth quarter]1B [ended yesterday.]1C (wsj0667) a9 ELABORATION a82 ATTRIBUTION 1A 1B 1C a82 ATTRIBUTION 1A a9 ELABORATION 1B 1C Figure 3: Two possible RST parses for a sentence.
---------------------------------------------------
P09-1075:6	41:186	The motivation for leaving aside segmenting were both practical  previous discourse parsing efforts (Soricut and Marcu, 2003; LeThanh et al., 2004) already provide alternatives for standalone segmenting tools  and scientific, namely, the greater need for improvements in labeling.
---------------------------------------------------
P09-1075:7	47:186	In accord with previous research (Soricut and Marcu, 2003; Reitter, 2003b; LeThanh et al., 2004), we turned all nary rhetorical relations into nested binary relations (a trivial graph transformation), resulting in more algorithmically manageable binary trees.
---------------------------------------------------
P09-1075:8	174:186	As suggested by previous research (Soricut and Marcu, 2003), these scores could likely be further improved with the use of better-performing segmenting algorithms.
---------------------------------------------------
P09-1075:9	156:186	For each corpus evaluation, the system is run twice: once using perfectly-segmented input (taken from the RST-DT), and once using the output of the SPADE segmenter (Soricut and Marcu, 2003).
---------------------------------------------------
D09-1024:10	115:194	The test set includes only sentences for which our English parser (Soricut and Marcu, 2003) could produce a parse tree, which effectively excluded a few very long sentences.
---------------------------------------------------
I05-6007:11	67:83	Still, the results and discussion from (Soricut and Marcu, 2003) provide some useful perspective on our results.
---------------------------------------------------
I05-6007:12	25:83	(Soricut and Marcu, 2003) In what follows, we first describe Attributions as they are understood in the RST Treebank project.
---------------------------------------------------
I05-6007:13	68:83	Soricut and Marcu (2003) evaluate their Discourse Parser under a variety of scenarios; the most favorable has human-corrected syntax trees and discourse segmentation.
---------------------------------------------------
I05-6007:14	23:83	The system performs dramatically better than the results reported in (Soricut and Marcu, 2003) for automatic identification of such relations, where the precision and recall were reported at below .76.
---------------------------------------------------
I05-6007:15	70:83	Soricut and Marcu (2003) note that human annotator agreement receives comparable f-scores, of .719 and .77 respectively.
---------------------------------------------------
I05-6007:16	65:83	+ a0 Prec Rec Basic 1497 215 0.874 Backwards 700 4 0.994 According-to 87 1 0.989 Total 2284 220 0.912 0.938 Figure 4: Breakdown of system results (Training corpus) + a0 Prec Rec Basic 193 33 0.854 Backwards 90 0 1.000 According-to 4 0 1.000 Total 286 33 0.897 0.994 Figure 5: Breakdown of system results (Test corpus) 60 5 Related Work Soricut and Marcu (2003) describe a Discourse Parser  a system that uses Penn Treebank syntax to identify intra-sentential discourse relations in the RST Treebank.
---------------------------------------------------
N07-1054:17	27:181	Marcu and Echihabi (2002) lter training instances based on Part-of-Speech (POS) tags, and Soricut and Marcu (2003) use syntactic features to identify sentence-internal RST structure.
---------------------------------------------------
N07-1054:18	23:181	Other work uses human-annotated corpora, such as the RST Bank (Carlson et al. , 2001), used by Soricut and Marcu (2003), the GraphBank (Wolf and Gibson, 2005), used by Wellner et al.
---------------------------------------------------
D07-1009:19	59:246	4Though statistical methods have been used to induce such trees (Soricut and Marcu, 2003), they are not used for ordering and other text-structuring tasks.
---------------------------------------------------
C04-1048:20	181:215	The most recent sentence-level discourse parser providing good results is SPADE, which is reported in (Soricut and Marcu 2003).
---------------------------------------------------
W05-0613:21	23:224	One exception is Marcus work (Marcu, 1997, 1999) (see also Soricut and Marcu (2003) for constructing discourse structures for individual sentences).
---------------------------------------------------
P09-2004:22	30:102	Most prior work on relation sense identification reports results obtained on data consisting of both explicit and implicit relations (Wellner et al., 2006; Soricut and Marcu, 2003).
---------------------------------------------------
P09-2004:23	40:102	2.2 Syntactic features Syntactic features have been extensively used for tasks such as argument identification: dividing sentences into elementary discourse units among which discourse relations hold (Soricut and Marcu, 2003; Wellner and Pustejovsky, 2007; Fisher and Roark, 2007; Elwell and Baldridge, 2008).
---------------------------------------------------
W04-2504:24	143:230	Furthermore, recent work also provides discourse annotated corpora with rhetorical relations (Carlson, et al. , 2003) and techniques for discourse paring for texts (Soricut and Marcu, 2003).
---------------------------------------------------
W06-2605:25	201:232	(Soricut and Marcu, 2003) reportthattheirSynDPparserachievedupto63.8FScoreonhuman-segmentedtestdata.
---------------------------------------------------
W06-2605:26	203:232	However, since the corpus, the size of training data and the set of rhetorical relations we have used differ from (Soricut and Marcu, 2003), a direct comparison cannot be made.
---------------------------------------------------
W06-2605:27	22:232	The task of discourse parsing can be divided into two disjoint sub-problems ((Soricut and Marcu, 2003) and (Polanyi et al. , 2004)).
---------------------------------------------------
W06-2605:28	26:232	It is trained on a lot fewer examples than the state of the art syntaxbased discourse parser (Soricut and Marcu, 2003).
---------------------------------------------------
W06-2605:29	8:232	Previous work on discourse parsing ((Soricut and Marcu, 2003) and (Forbes et al. , 2001)) have focused on syntactic and lexical features only.
---------------------------------------------------
W08-2226:30	23:118	It then applies syntax-based discourse parsing rules (such as Soricut and Marcu (2003)) to reduce coordinate, subordinate, and relative clauses into coindexed, simpler sentence parses headed by single verbal relations.
---------------------------------------------------
W08-2226:31	35:118	Soricut and Marcu (2003) presented a statistical system that automatically produces an analysis of the rhetorical structure that holds between sets of sentences or clauses at the paragraph level.
---------------------------------------------------
P07-1075:32	97:222	We describe the nodes and edges of T l separately for each type, because their representations are different: 1) The nodes of discourse tree T c consist of clauses [Clause 1, , Clause Ncl ]; and their relation edges are obtained from the Spade system described in Soricut and Marcu (2003).
---------------------------------------------------
N04-1015:33	165:174	By automatically annotating a large corpus of texts with discourse relations via a rhetorical parser (Marcu, 1997; Soricut and Marcu, 2003), we may be able to incorporate domain-independent relationships into the transition structure of our content models.
---------------------------------------------------
W06-1317:34	21:188	Within Rhetorical Structure Theory (RST), Soricut and Marcu (2003) have developed two 117 probabilistica5 models for identifying clausal elementary discourse units and generating discourse trees at the sentence level.
---------------------------------------------------
W04-1003:35	37:155	SEE allowed the judges to step through predefined units of the model summary (elementary discourse units/EDUs) (Soricut and Marcu, 2003) and for each unit of that summary, mark the sentences in the peer summary that expressed [all(4), most(3), some(2), hardly any(1) or none(0)] of the content in the current model summary unit.
---------------------------------------------------
C04-1007:36	10:191	While these tasks have been dealt with quite well for small structures (i.e. on clause and sentence level) (Soricut and Marcu, 2003), many of these approaches cannot be applied directly to higherlevel structures (e.g. on multi-sentence and interparagraph level) because they rely nearly exclusively on cue phrases, which are much less useful for large structures (Marcu, 2000, p. 129).
---------------------------------------------------
H05-1033:37	14:234	Even though discourse parsing at the document-level still poses a significant challenge to data-driven methods, sentence-level discourse models (e.g. , Soricut and Marcu, 2003) trained on the RST-DT have attained accuracies comparable to human performance.
---------------------------------------------------
H05-1033:38	150:234	We also applied Spade5, Soricut and Marcus (2003) sentence-level discourse parser (see Section 2) to our test set.
---------------------------------------------------
H05-1033:39	35:234	Soricut and Marcu (2003) introduce a syntax-based sentencelevel discourse parser, which consists of two components: a statistical segmentation model and a parser working on the output of the segmenter.
---------------------------------------------------
H05-1033:40	24:234	For example, Soricut and Marcu (2003) show that perfect discourse segmentation delivers an error reduction of 29% in the performance of their discourse parser.
---------------------------------------------------
H05-1033:41	203:234	We applied the same strategy to derive compressed sentences from the output of Spade (Soricut and Marcu, 2003), and also produced human compressions.
---------------------------------------------------
H05-1033:42	30:234	We employ lexical and low-level syntactic information (e.g. , parts of speech, syntactic chunks) and show that the performance of our discourse chunker on the two subtasks (mentioned above) is comparable to that of a stateof-the-art sentence-level discourse parser (Soricut and Marcu, 2003).
---------------------------------------------------
N06-1027:43	45:204	Most of the current work on discourse processing focuses on sentence-level text organization (Soricut and Marcu, 2003) or the intermediate step (Sporleder and Lapata, 2005).
---------------------------------------------------
D09-1130:44	244:245	In the future, we plan to explore other related problems such as adjacency pairs (Levinson, 1983) and discourse parsing (Soricut and Marcu, 2003) for large-scale online forum data.
---------------------------------------------------
P07-1062:45	50:192	For those trials, the 951 sentence subset from Soricut and Marcu (2003) is used.
---------------------------------------------------
P07-1062:46	173:192	Our experiments also confirm that increased segmentation accuracy yields significantly better discourse parsing accuracy, as previously shown to be the case when providing reference segmentations to a parser (Soricut and Marcu, 2003).
---------------------------------------------------
P07-1062:47	34:192	Human agreement for this segmentation task is quite high, with agreement between two annotators at an F-score of 98.3 for unlabeled segmentation (Soricut and Marcu, 2003).
---------------------------------------------------
P07-1062:48	54:192	Hence Soricut and Marcu (2003) evaluate with respect to sentence internal segmentationboundaries, i.e., withindicesj suchthat0<j<k for a sentence of length k. Let g be the number of sentence-internal segmentation boundaries in the goldstandard,tthenumberofsentence-internalsegmentation boundaries in the system output, and m the number of correct sentence-internal segmentation boundaries in the system output.
---------------------------------------------------
P07-1062:49	15:192	Sporleder and Lapata (2005) also used the RST Treebank as training data for data-driven discourse parsing algorithms, though their focus, in contrast to Soricut and Marcu (2003), was to avoid contextfree parsing and rely exclusively on features in their model that could be derived via finite-state chunkers andtaggers.
---------------------------------------------------
P07-1062:50	12:192	Using the RST Treebank as training and evaluation data, Soricut and Marcu (2003) demonstrated that their automatic sentence-level discourse parsing system could achieve near-human levels of accuracy, if it was provided with manual segmentations and manual parse trees.
---------------------------------------------------
P07-1062:51	43:192	Soricut and Marcu (2003) 2A small number of document final parentheticals are in the RST-DT and not in the Penn WSJ Treebank, which our alignment approach takes into account.
---------------------------------------------------
P07-1062:52	161:192	3.2 Discourse parsing It has been shown that accurate discourse segmentation within a sentence greatly improves the overall parsing accuracy to near human levels (Soricut and Marcu, 2003).
---------------------------------------------------
P07-1062:53	129:192	Segment boundary accuracy is for sentence internal boundaries only, following Soricut and Marcu (2003).
---------------------------------------------------
P07-1062:54	60:192	2.3 Baseline SPADE setup The publicly available SPADE package, which encodes the approach in Soricut and Marcu (2003), is taken as the baseline for this paper.
---------------------------------------------------
P07-1062:55	155:192	493 Segmentation Unlabeled Nuc/Sat SPADE 76.9 70.2 Classifier: Full finite state 78.1 71.1 Classifier: All features 83.5 76.1 Table 3: Discourse parsing results on the 951 sentence Soricut and Marcu (2003) evaluation set, using SPADE for parsing, and various methods for segmentation.
---------------------------------------------------
N04-1020:56	242:249	This result is in agreement with Soricut and Marcu (2003) who find that syntax trees encode sufficient information to enable accurate derivation of discourse relations.
---------------------------------------------------
W04-2322:57	116:167	Soricut and Marcu (2003) also build up RST sentential trees to use in discourse parsing.
---------------------------------------------------
P09-1077:58	25:262	Soricut and Marcu (2003) address the task of 683 parsing discourse structures within the same sentence.
---------------------------------------------------
N07-3006:59	46:84	Our model was trained and tested on RST-DT (2002) and achieves a performance of up to 86.12% F-Score, which is comparable to Soricut and Marcu (2003).
---------------------------------------------------
N07-3006:60	44:84	In the past, the problem of sentence level discourse segmentation has been tackled using both symbolic methods (Polanyi et al. , 2004; Huong et al. , 2004) as well as statistical models (Soricut and Marcu, 2003; Marcu, 2000) that have exploited syntactic and lexical features.
---------------------------------------------------
P09-2020:61	7:93	Since segmentationisthefirststageofdiscourseparsing, quality discourse segments are critical to building quality discourse representations (Soricut and Marcu, 2003).
---------------------------------------------------
P09-2020:62	19:93	2 Related Work Soricut and Marcu (2003) construct a statistical discourse segmenter as part of their sentence-level discourse parser (SPADE), the only implementation available for our comparison.
---------------------------------------------------
