P09-1104:1	82:219	A simple loss-augmented learning procedure is the margin infused relaxed algorithm (MIRA) (Crammer et al., 2006).
---------------------------------------------------
P09-1104:2	140:219	We use posteriors from two jointly estimated HMM models to make pruning decisions during ITG inference (Liang et al., 2006).
---------------------------------------------------
P09-1104:3	11:219	Past work on discriminative word alignment has focused on the family of at-most-one-to-one matchings (Melamed, 2000; Taskar et al., 2005; Moore et al., 2006).
---------------------------------------------------
P09-1104:4	40:219	The alignment error rate (AER) is given by, AER(a,s,p) = 1|as|+|ap||a|+|s| 2.1 1-to-1 Matchings The class of at most 1-to-1 alignment matchings, A1-1, has been considered in several works (Melamed, 2000; Taskar et al., 2005; Moore et al., 2006).
---------------------------------------------------
P09-1104:5	167:219	6.1 French Hansards Results The French Hansards data are well-studied data sets for discriminative word alignment (Taskar et al., 2005; Cherry and Lin, 2006; Lacoste-Julien et al., 2006).
---------------------------------------------------
J07-3002:6	162:164	F-Measure with an appropriate setting of  will be useful during the development process of new alignment models, or as a maximization criterion for discriminative training of alignment models (Cherry and Lin 2003; Ayan, Dorr, and Monz 2005; Ittycheriah and Roukos 2005; Liu, Liu, and Lin 2005; Fraser and Marcu 2006; Lacoste-Julien et al. 2006; Moore, Yih, and Bode 2006).
---------------------------------------------------
W07-1520:7	12:110	Because of its central role in building machine translation systems and because of the complexity of the task, sub-sentential alignment of parallel corpora continues to be an active area of research (e.g. , Moore et al. , 2006; Fraser and Marcu, 2006), and this implies a continuing demand for manually created or human-verified gold standard alignments for development and evaluation purposes.
---------------------------------------------------
W08-0306:8	74:125	In general, Agold / Acandidates; following (Collins, 2000) and (Charniak and Johnson, 2005) for parse reranking and (Liang et al., 2006) for translation reranking, we define Aoracle as alignment in Acandidates that is most similar to Agold.8 We update each feature weight i as follows: i = i + hAoraclei hA1-besti .9 Following (Moore, 2005), after each training pass, we average all the feature weight vectors seen during the pass, and decode the discriminative training set using the vector of averaged feature weights.
---------------------------------------------------
W08-0306:9	12:125	The low precision of GIZA++ union alignments poses a particular problem for syntax-based rule extraction algorithms such as (Quirk et al., 2005; Galley et al., 2006; Huang et al., 2006; Liu et al., 2006): if the incorrect links violate syntactic correspondences, they force the rule extraction algorithm to extract rules that are large in size, few in number, and poor in generalization ability.
---------------------------------------------------
W08-0306:10	8:125	GIZA++ union alignments have been used in the state-of-the-art syntax-based statistical MT system described in (Galley et al., 2006) and in the hierarchical phrase-based system Hiero (Chiang, 2007).
---------------------------------------------------
W08-0306:11	81:125	9(Liang et al., 2006) report that, for translation reranking, such local updates (towards the oracle) outperform bold updates (towards the gold standard).
---------------------------------------------------
W08-0306:12	24:125	1.2 Related Work Recently, discriminative methods for alignment have rivaled the quality of IBM Model 4 alignments (Liu et al., 2005; Ittycheriah and Roukos, 2005; Taskar et al., 2005; Moore et al., 2006; Fraser and Marcu, 2007b).
---------------------------------------------------
W08-0306:13	96:125	Using these alignments, which we refer to as GIZA++ union + link deletion, we train a syntax-based translation system similar to that described in (Galley et al., 2006).
---------------------------------------------------
W08-0306:14	32:125	(Lopez and Resnik, 2005) and (Denero and Klein, 2007) modify the distortion model of the HMM alignment model (Vogel et al., 1996) to reflect tree distance rather than string distance; (Cherry and Lin, 2006) modify an ITG aligner by introducing a penalty for induced parses that violate syntactic bracketing constraints.
---------------------------------------------------
W07-0711:15	58:235	In practice, we can bucket the distortion parameters {c(d)} into a few buckets as implemented in (Liang et al. , 2006).
---------------------------------------------------
W07-0711:16	8:235	The classical approaches to word alignment are based on IBM models 1-5 (Brown et al. , 1994) and the HMM based alignment model (Vogel et al. , 1996) (Och and Ney, 2000a, 2000b), while recently discriminative approaches (Moore, 2006) and syntax based approaches (Zhang and Gildea, 2005) for word alignment are also studied.
---------------------------------------------------
W07-0711:17	61:235	As suggested in (Liang et al. , 2006), we also use two separate sets of distortion parameters for transitioning into the first state, and for transitioning out of the last state, respectively.
---------------------------------------------------
N07-2053:18	23:103	In this work, we use a discriminatively trained word aligner (Moore et al. , 2006) that has state of the art performance.
---------------------------------------------------
N07-2053:19	99:103	The original approach (Marcu and Wong, 2002) was limited due to computational constraints but recent work (DeNero et al. , 2006; Birch et al. , 2006) has improved the efficiency by using word alignments as constraints on the set of possible phrase pairs.
---------------------------------------------------
N07-2053:20	36:103	(2006), modified from (Koehn et al. , 2003), which is an average of pairwise word translation probabilities.
---------------------------------------------------
D07-1006:21	166:193	(Liang et al. , 2006) showed how to train two HMM models, a 1-to-N model and a M-to-1 model, to agree in predicting all of the links generated, resulting in a 1-to-1 alignment with occasional rare 1to-N or M-to-1 links.
---------------------------------------------------
D07-1006:22	175:193	(Lacoste-Julien et al. , 2006) created a discriminative model able to model 1-to-1, 1-to2 and 2-to-1 alignments for which the best results were obtained using features based on symmetric HMMs trained to agree, (Liang et al. , 2006), and intersected Model 4.
---------------------------------------------------
D07-1006:23	163:193	(Och and Ney, 2003) invented heuristic symmetriza57 FRENCH/ENGLISH ARABIC/ENGLISH SYSTEM F-MEASURE ( = 0.4) BLEU F-MEASURE ( = 0.1) BLEU GIZA++ 73.5 30.63 75.8 51.55 (FRASER AND MARCU, 2006B) 74.1 31.40 79.1 52.89 LEAF UNSUPERVISED 74.5 72.3 LEAF SEMI-SUPERVISED 76.3 31.86 84.5 54.34 Table 3: Experimental Results tion of the output of a 1-to-N model and a M-to-1 model resulting in a M-to-N alignment, this was extended in (Koehn et al. , 2003).
---------------------------------------------------
D07-1006:24	178:193	(Moore et al. , 2006) introduced a discriminative model of 1-to-N and M-to-1 alignments, and similarly to (Lacoste-Julien et al. , 2006) the best results were obtained using HMMs trained to agree and intersected Model 4.
---------------------------------------------------
P08-4006:25	17:76	2 Word alignment visualization Over the years, numerous tools for the visualization and creation of word alignments have been developed (e.g., Melamed, 1998; Smith and Jahr, 2000; Ahrenberg et al., 2002; Rassier and Pedersen, 2003; Daume; Tiedemann; Hwa and Madnani, 2004; Lambert, 2004; Tiedemann, 2006).
---------------------------------------------------
P08-4006:26	12:76	Consequently, considerable effort has gone into devising and improving automatic word alignment algorithms, and into evaluating their performance (e.g., Och and Ney, 2003; Taskar et al., 2005; Moore et al., 2006; Fraser and Marcu, 2006, among many others).
---------------------------------------------------
W08-0303:27	135:239	We follow (Gao et al., 2006; Suzuki et al., 2006) and approximate the metrics using the sigmoid function.
---------------------------------------------------
W08-0303:28	147:239	The POS-tags were generated by the Brill-Tagger (Brill, 1995) and the FreeLing-Tagger (Asterias et al., 2006) for the English and the Spanish text respectively.
---------------------------------------------------
W08-0303:29	42:239	The structure of the CRF is described by a factored graph like it was done, for example, in (Lan et al., 2006).
---------------------------------------------------
W08-0303:30	17:239	In recent years several authors (Moore et al., 2006; Lacoste-Julien et al., 2006; Blunsom and Cohn, 2006) proposed discriminative word alignment frameworks and showed that this leads to improved alignment quality.
---------------------------------------------------
D09-1106:31	230:235	Another interesting direction is applying our approach to extracting translation rules with hierarchical structures such as hierarchical phrases (Chiang, 2007) and tree-to-string rules (Galley et al., 2006; Liu et al., 2006).
---------------------------------------------------
D09-1106:32	10:235	Word-aligned corpora have been found to be an excellent source for translation-related knowledge, not only for phrase-based models (Och and Ney, 2004; Koehn et al., 2003), but also for syntax-based models (e.g., (Chiang, 2007; Galley et al., 2006; Shen et al., 2008; Liu et al., 2006)).
---------------------------------------------------
D09-1106:33	14:235	In contrast, discriminative methods (e.g., (Moore et al., 2006; Taskar et al., 2005; Liu et al., 2005; Blunsom and Cohn, 2006)) have the freedom to define arbitrary feature functions that describe various characteristics of an alignment.
---------------------------------------------------
W07-0715:34	14:155	(2006) tried the Marcu and Wong model constrained by a word alignment and also found that Koehn, et al.?s model worked better, with the advantage of the standard model increasing as more features were added to the overall translation model.
---------------------------------------------------
W07-0715:35	90:155	These 500,000 sentence pairs were word-aligned using a state-ofthe-art word-alignment method (Moore et al. , 2006).
---------------------------------------------------
