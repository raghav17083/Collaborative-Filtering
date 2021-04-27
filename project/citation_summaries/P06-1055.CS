D09-1119:1	41:258	This was recently followed by (Matsuzaki et al., 2005; Petrov et al., 2006) who introduce state-of-the-art nearly unlexicalized PCFG parsers.
---------------------------------------------------
P08-1109:2	163:173	Following up on their previous work (Petrov et al., 2006) on grammar splitting, they do discriminative 965 S S NP PRP He VP VBZ adds NP DT This VP VBZ is RB nt NP NP CD 1987 VP VBN revisited S NP PRP He VP VBZ adds S NP DT This VP VBZ is RB nt NP CD 1987 VP VBN revisited S NP PRP He VP VBZ adds S NP DT This VP VBZ is RB nt NP NP CD 1987 VP VBN revisited (a) generative output (b) feature-based discriminative output (c) gold parse Figure 3: Example output from our generative and feature-based discriminative models, along with the correct parse.
---------------------------------------------------
P08-1109:3	97:173	We viewed this as equivalent to the more elaborate, smoothed unknown word models that are common in many PCFG parsers, such as (Klein and Manning, 2003; Petrov et al., 2006).
---------------------------------------------------
N09-1062:4	162:209	For comparison, we trained the Berkeley splitmerge (SM) parser (Petrov et al., 2006) on the same data and decoded using the Viterbi algorithm (MPD) and expected rule count (MER a.k.a. MAX-RULESUM).
---------------------------------------------------
N09-1062:5	38:209	Our work differs from these previous approaches in that we explicitly model a prior over grammars within a Bayesian framework.4 Models of grammar refinement (Petrov et al., 2006; Liang et al., 2007; Finkel et al., 2007) also aim to automatically learn latent structure underlying treebanked data.
---------------------------------------------------
N09-2054:6	24:76	Rather than explicit annotation, we could use latent annotations to split the POS tags, similarly to the introduction of latent annotations to PCFG grammars (Matsuzaki et al., 2005; Petrov et al., 2006).
---------------------------------------------------
N09-2054:7	7:76	Building upon the large body of research to improve tagging performance for various languages using various models (e.g., (Thede and Harper, 1999; Brants, 2000; Tseng et al., 2005b; Huang et al., 2007)) and the recent work on PCFG grammars with latent annotations (Matsuzaki et al., 2005; Petrov et al., 2006), we will investigate the use of fine-grained latent annotations for Chinese POS tagging.
---------------------------------------------------
N09-2054:8	29:76	i(ax) = summationdisplay yp(by|ax)p(wi+1|by)j+1(by) In the E step, the posterior probabilities of cooccurrence events can be computed as: p(ax,by|w)  i(ax)p(by|ax)i+1(by) p(ax,wi|w)  i(ax)i(ax) In the M step, the above posterior probabilities are used as weighted observations to update the transition and emission probabilities2: p(by|ax) = c(ax,by)/ summationdisplay by c(ax,by) p(w|ax) = c(ax,w)/ summationdisplay wc(ax,w) A hierarchical split-and-merge method, similar to (Petrov et al., 2006), is used to gradually increase the number of latent annotations while allocating them adaptively to places where they would produce the greatest increase in training likelihood (e.g., we observe heavy splitting in categories such as NN (normal noun) and VV (verb), that cover a wide variety of words, but only minimal splitting in categories like IJ (interjection) and ON (onomatopoeia)).
---------------------------------------------------
P07-1022:9	14:137	For example, incremental CFG parsing algorithms can be used with the CFGs produced by this transform, as can the Inside-Outside estimation algorithm (Lari and Young, 1990) and more exotic methods such as estimating adjoined hidden states (Matsuzaki et al. , 2005; Petrov et al. , 2006).
---------------------------------------------------
P07-1035:10	18:171	Hence, state-of-the-art parsers either supplement the part-of-speech (POS) tags with the lexical forms themselves (Collins, 2003; Charniak, 2000), manually split the tagset into a finer-grained one (Klein and Manning, 2003a), or learn finer grained tag distinctions using a heuristic learning procedure (Petrov et al. , 2006).
---------------------------------------------------
D07-1058:11	199:308	The exact relation of P-DOP to other DOP models, including S-DOP (Bod, 2003), Backoff-DOP (Simaan and Buratto, 2003), DOP* (Zollmann and Simaan, 2005) and ML-DOP (Bod, 2006; based on Expectation Maximization) and not dissimilar automatic enrichment models such as (Petrov et al. , 2006), remains a topic for future work.
---------------------------------------------------
D07-1058:12	16:308	Moreover, as P-DOP is formulated as an enrichment of the treebank Probabilistic Context-free Grammar (PCFG), it allows for much easier comparison to alternative approaches to statistical parsing (Collins, 1997; Charniak, 1997; Johnson, 1998; Klein and Manning, 2003; Petrov et al. , 2006).
---------------------------------------------------
D07-1058:13	21:308	Interesting recent work has focused on the automatic induction of enrichments (Matzuzaki et al. , 2005; Prescher, 2005), leading to extremely accurate parsers (Petrov et al. , 2006).
---------------------------------------------------
D07-1058:14	67:308	PCFG enrichment models (Klein and Manning, 2003; Schmid, 2006) split (and merge) nonterminals; in automatic enrichment methods (Prescher, 2005; Petrov et al. , 2006) these transformations are performed so as to maximize data likelihood (under some constraints).
---------------------------------------------------
E09-1080:15	32:186	2However, the authors approach of using EM for inducing latent information in treebanks has led to extremely accurate constituency parsers, that neither make use of nor produce headedness information; see (Petrov et al., 2006) then describe our evaluations of these algorithms.
---------------------------------------------------
W08-1005:16	25:154	2 Latent Variable Parsing In latent variable parsing (Matsuzaki et al., 2005; Prescher, 2005; Petrov et al., 2006), we learn rule probabilities on latent annotations that, when marginalized out, maximize the likelihood of the unannotated training trees.
---------------------------------------------------
P07-2052:17	31:109	splitting tags (Matsuzaki et al. , 2005; Petrov et al. , 2006).
---------------------------------------------------
P07-2052:18	9:109	Unlexicalized parsers, on the other hand, achieved accuracies almost equivalent to those of lexicalized parsers (Klein and Manning, 2003; Matsuzaki et al. , 2005; Petrov et al. , 2006).
---------------------------------------------------
D08-1091:19	219:223	methods like split-and-merge (Petrov et al., 2006).
---------------------------------------------------
D08-1091:20	169:223	7.3 Efficiency Petrov and Klein (2007) demonstrates how the idea of coarse-to-fine parsing (Charniak et al., 1998; Charniak et al., 2006) can be used in the context of latent variable models.
---------------------------------------------------
D08-1091:21	9:223	1 Introduction In latent variable approaches to parsing (Matsuzaki et al., 2005; Petrov et al., 2006), one models an observed treebank of coarse parse trees using a grammar over more refined, but unobserved, derivation trees.
---------------------------------------------------
W09-1008:22	95:224	(Petrov et al., 2006) (hereafter BKY) overcomes this problem by using the following algorithm: given a PCFG G0 induced from the treebank, iteratively create n grammars G1 Gn (with n = 5 in practice), where each iterative step is as follows :  SPLIT Create a new grammar Gi from Gi1 by splitting every non terminal of Gi in two new symbols.
---------------------------------------------------
W09-1008:23	216:224	We observe however that (Petrov et al., 2006)s semi-supervised learning procedure is not fully optimal since a manual refinement of the treebank labelling turns out to improve the parsing results.
---------------------------------------------------
W09-1008:24	211:224	6 Conclusion This paper reports results in statistical parsing for French with both unlexicalized (Petrov et al., 2006) and lexicalized parsers.
---------------------------------------------------
W09-1008:25	4:224	We compare, for French, a supervised lexicalized parsing algorithm with a semi-supervised unlexicalized algorithm (Petrov et al., 2006) along the lines of (Crabb and Candito, 2008).
---------------------------------------------------
W09-1008:26	63:224	Solving this first methodological issue, has led to solutions dubbed hereafter as unlexicalized statistical parsing (Johnson, 1998; Klein and Manning, 2003a; Matsuzaki et al., 2005; Petrov et al., 2006).
---------------------------------------------------
W09-1008:27	23:224	This leads to 49 methods that use semi-supervised techniques on a treebank-infered grammar backbone, such as (Matsuzaki et al., 2005; Petrov et al., 2006).
---------------------------------------------------
W09-1008:28	144:224	2000 4000 6000 8000 10000 76 78 80 82 84 86 88 Number of training sentences Fscore Bikel Berkeley Figure 3: Parsing Learning curve on FTB with CCtagset, in perfect-tagging This potential increase for BKY results if we had more French annotated data is somehow confirmed by the higher results reported for BKY training on the Penn Treebank (Petrov et al., 2006) : F1=90.2.
---------------------------------------------------
W09-1008:29	117:224	We 3(Petrov et al., 2006) obtain an F-score=90.1 for sentences of less than 40 words.
---------------------------------------------------
D07-1072:30	26:303	Unlexicalized methods refine the grammar in a more conservative fashion, splitting each non-terminal or pre-terminal symbol into a much smaller number of subsymbols (Klein and Manning, 2003; Matsuzaki et al. , 2005; Petrov et al. , 2006).
---------------------------------------------------
P08-4003:31	29:54	BART also supports using the Berkeley parser (Petrov et al., 2006), yielding an easy-to-use Java-only solution.
---------------------------------------------------
P09-1008:32	5:235	We report the results of topological field parsing of German using the unlexicalized, latent variable-based Berkeley parser (Petrov et al., 2006) Without any languageor model-dependent adaptation, we achieve state-of-the-art results on the TuBa-D/Z corpus, and a modified NEGRA corpus that has been automatically annotated with topological fields (Becker and Frank, 2002).
---------------------------------------------------
P09-1008:33	10:235	We report the results of parsing German using the unlexicalized, latent variable-based Berkeley parser (Petrov et al., 2006).
---------------------------------------------------
P09-1008:34	89:235	 (b) Translation: One must understand, he said, that these minorities have been massively threatened by the Nazis for a long time. 3 A Latent Variable Parser For our experiments, we used the latent variablebased Berkeley parser (Petrov et al., 2006).
---------------------------------------------------
P08-2026:35	51:102	However, several very good current parsers were not available when this paper was written (e.g., the Berkeley Parser (Petrov et al., 2006)).
---------------------------------------------------
C08-1071:36	7:236	1 Introduction Supervised statistical parsers attempt to capture patterns of syntactic structure from a labeled set of examples for the purpose of annotating new sentences with their structure (Bod, 2003; Charniak and Johnson, 2005; Collins and Koo, 2005; Petrov et al., 2006; Titov and Henderson, 2007).
---------------------------------------------------
D07-1094:37	70:175	We approximate the loss in data likelihood for a merge s1s2  swiththefollowinglikelihoodratio(Petrov et al. , 2006): (s1 s2  s) = productdisplay sequences productdisplay t Pt(x,y) P(x,y) . Here P(x,y) is the joint likelihood of an emission sequence x and associated state sequence y. This quantity can be recovered from the forward and backward probabilities using P(x,y) = summationdisplay s:pi(s)=yt t(s)t(s).
---------------------------------------------------
D07-1094:38	111:175	3Following previous work with PCFGs (Petrov et al. , 2006), we experimented with smoothing the substates towards each other to prevent overfitting, but we were unable to achieve any performance gains.
---------------------------------------------------
D07-1094:39	17:175	For example in the domain of syntactic parsing with probabilistic context-free grammars (PCFGs), a surprising recent result is that automatically induced grammar refinements can outperform sophisticated methods which exploit substantial manually articulated structure (Petrov et al. , 2006).
---------------------------------------------------
N09-1025:40	112:173	For the source-side syntax features, we used the Berkeley parser (Petrov et al., 2006) to parse the Chinese side of both sets.
---------------------------------------------------
D09-1088:41	191:205	A promising strategy then would be to smooth or split-and-merge (Petrov et al., 2006)) RR-based models rather than to add an elaborate smoothing component to configurationally-based HD models.
---------------------------------------------------
N07-1051:42	214:232	Space does not permit a thorough exposition of our analysis, but as in the case of English (Petrov et al. , 2006), the learned subcategories exhibit interesting linguistic interpretations.
---------------------------------------------------
N07-1051:43	8:232	Previous work has shown that high-quality unlexicalized PCFGs can be learned from a treebank, either by manual annotation (Klein and Manning, 2003) or automatic state splitting (Matsuzaki et al. , 2005; Petrov et al. , 2006).
---------------------------------------------------
D08-1018:44	201:299	However, there are state-of-the-art unlexicalized parsers (Klein and Manning, 2003b; Petrov et al., 2006), to which we believe our binarization can be applied.
---------------------------------------------------
D09-1161:45	9:255	In general, they can be divided into two major categories, namely lexicalized models (Collins 1997, 1999; Charniak 1997, 2000) and un-lexicalized models (Klein and Manning 2003; Matsuzaki et al. 2005; Petrov et al. 2006; Petrov and Klein 2007).
---------------------------------------------------
D09-1161:46	81:255	The latent-annotation model (Matsuzaki et al. 2005; Petrov et al. 2006) is one of the most effective un-lexicalized models.
---------------------------------------------------
W07-2219:47	103:344	3.1 A Note on State-Splits Recent studies (Klein and Manning, 2003; Matsuzaki et al. , 2005; Prescher, 2005; Petrov et al. , 2006) suggest that category-splits help in enhancing the performance of treebank grammars, and a previous study on MH (Tsarfaty, 2006) outlines specific POS-tags splits that improve MH parsing accuracy.
---------------------------------------------------
D09-1087:48	208:227	We observe that many of the rule parameters of the grammar trained on WSJ training data alone have zero probabilities (rules with extremely low probabilities are also filtered to zero), as was also pointed out in (Petrov et al., 2006).
---------------------------------------------------
D09-1087:49	34:227	2 Parsing Model The Berkeley parser (Petrov et al., 2006; Petrov and Klein, 2007) is an efficient and effective parser that introduces latent annotations (Matsuzaki et al., 2005) to refine syntactic categories to learn better PCFG grammars.
---------------------------------------------------
D09-1087:50	6:227	1 Introduction There is an extensive research literature on building high quality parsers for English (Collins, 1999; Charniak, 2000; Charniak and Johnson, 2005; Petrov et al., 2006), however, models for parsing other languages are less well developed.
---------------------------------------------------
D07-1014:51	132:189	More recently, EM has been used to learn hidden variables in parse trees; these can be head-childannotations(ChiangandBikel, 2002), latent head features (Matsuzaki et al. , 2005; Prescher, 2005; Dreyer and Eisner, 2006), or hierarchicallysplit nonterminal states (Petrov et al. , 2006).
---------------------------------------------------
D07-1014:52	152:189	6 Discussion Noting that adding latent features to nonterminals in unlexicalized context-free parsing has been very successful (Chiang and Bikel, 2002; Matsuzaki et al. , 2005; Prescher, 2005; Dreyer and Eisner, 2006; Petrov et al. , 2006), we were surprised not to see a 3Czech experiments were not done, since the number of features (more than 14 million) was too high to multiply out by clusters.
---------------------------------------------------
D08-1012:53	37:215	For example, in monolingual parsing, posterior pruning methods (Goodman, 1997; Charniak et al., 2006; Petrov and Klein, 2007) have led to greater speedups than their more cautious A* analogues (Klein and Manning, 2003; Haghighi et al., 2007), though at the cost of guaranteed optimality.
---------------------------------------------------
D08-1012:54	21:215	We demonstrate that likelihood-based hierarchical EM training (Petrov et al., 2006) and cluster-based language modeling methods (Goodman, 2001) are superior to both rank-based and random-projection methods.
---------------------------------------------------
P08-1068:55	186:218	Previous research in this area includes several models which incorporate hidden variables (Matsuzaki et al., 2005; Koo and Collins, 2005; Petrov et al., 2006; Titov and Henderson, 2007).
---------------------------------------------------
