P08-1069:1	23:224	and Gildea, 2007; Zhang et al., 2006; Gildea, Satta, and Zhang, 2006).
---------------------------------------------------
N09-1061:2	9:186	Not only is this beneficial in terms of parsing complexity, but smaller rules can also improve a translation models ability to generalize to new data (Zhang et al., 2006).
---------------------------------------------------
D09-1038:3	43:178	The experimental results show that our method outperforms the synchronous binarization method (Zhang et al., 2006) with over 0.8 BLEU scores on both NIST 2005 and NIST 2008 Chinese-to-English evaluation data sets.
---------------------------------------------------
D09-1038:4	17:178	Given the following SCFG rule:      VP      VB  NP     JJR  ,                VB  NP  will be  JJR we can obtain a set of equivalent binary rules using the synchronous binarization method (Zhang et al., 2006)  as follows:         VP  V1  JJR ,   V1  JJR             V1  VB  V2 ,   VB  V2         V2  NP   ,   NP  will be This binarization is shown with the solid lines as binarization (a) in Figure 1.
---------------------------------------------------
D09-1038:5	123:178	4.2 Binarization Schemes Besides the baseline (Zhang et al., 2006) and iterative cost reduction binarization methods, we also perform right-heavy and random synchronous binarizations for comparison.
---------------------------------------------------
D09-1038:6	51:178	Although this method is comparatively easy to be implemented, it just achieves the same performance as the synchronous binarization method (Zhang et al., 2006) for syntaxbased SMT systems.
---------------------------------------------------
D09-1038:7	28:178	V B N P  J J R ( a ) ( b ) V 2 V 1 V 2 ' V 1 ' V P V B N P w ill b e J J R  Figure 1: Two different binarizations (a) and (b) of the same SCFG rule distinguished by the solid lines and dashed lines                         ( W e  h o p e  t h e  s i t u a t i o n  w i l l  b e  b e t t e r  . )           N P        J J R     d e c o d i n g m a t c h  8 7 4  r u l e s m a t c h  6 2  r u l e s c o m p e t i n g  e d g e s :  8 0 1 c o m p e t i n g  e d g e s :  5 7 Figure 2: Edge competitions caused by different binarizations  The edge competition problem for SMT decoding is not addressed in previous work (Zhang et al., 2006; Huang, 2007) in which each SCFG rule is binarized in a fixed way.
---------------------------------------------------
D09-1038:8	22:178	Generally, two edges can be re-combined if they satisfy the following two constraints:  1) the LHS (left-hand side) nonterminals are identical and the sub-alignments are the same (Zhang et al., 2006); and 2) the boundary words 1  on both sides of the partial translations are equal between the two edges (Chiang, 2007).
---------------------------------------------------
D09-1038:9	48:178	A synchronous 363 binarization method is proposed in (Zhang et al., 2006) whose basic idea is to build a left-heavy binary synchronous tree (Shapiro and Stephens, 1991) with a left-to-right shift-reduce algorithm.
---------------------------------------------------
D09-1038:10	90:178	The time complexity of the CKY-based binarization algorithm is  (n3), which is higher than that of the linear binarization such as the synchronous binarization (Zhang et al., 2006).
---------------------------------------------------
D09-1038:11	95:178	Iterative cost reduction algorithm Input: An SCFG  Output: An equivalent binary SCFG    of  1: Function ITERATIVECOSTREDUCTION( ) 2:         0 3:   for each    0do 4:         ( ) =     ,  0 5:   while  (  ) does not converge do 6:        for each      do 7:             [   ]        (  ) 8:            for each     (  ) do 9:                for each        ,     do 10:                              1 11:           (  )   CKYBINARIZATION(  ,  ) 12:                [   ]    (  ) 13:          for each     (  ) do 14:              for each        ,     do 15:                             + 1 16: return   In the iterative cost reduction algorithm, we first obtain an initial binary SCFG  0 using the synchronous binarization method proposed in (Zhang et al., 2006).
---------------------------------------------------
D09-1038:12	53:178	3 Synchronous Binarization Optimization by Cost Reduction As discussed in Section 1, binarizing an SCFG in a fixed (left-heavy) way (Zhang et al., 2006) may lead to a large number of competing edges and consequently high risk of making search errors.
---------------------------------------------------
D09-1038:13	122:178	The baseline system is based on the synchronous binarization (Zhang et al., 2006).
---------------------------------------------------
W07-0405:14	14:147	Synchronous binarization (Zhang et al. , 2006) solves this problem by simultaneously binarizing both source and target-sides of a synchronous rule, making sure of contiguous spans on both sides whenever possible.
---------------------------------------------------
W07-0405:15	29:147	Decoding with an SCFG (e.g. , translating from Chinese to English using the above grammar) can be cast as a parsing problem (see Section 3 for details), in which case we need to binarize a synchronous rule with more than two nonterminals to achieve polynomial time algorithms (Zhang et al. , 2006).
---------------------------------------------------
W07-0405:16	43:147	This representation, being contiguous on both sides, successfully reduces the decoding complexity to a low polynomial and significantly improved the search quality (Zhang et al. , 2006).
---------------------------------------------------
W07-0405:17	36:147	Intuitively speaking, the gaps on the target-side will lead to exponential complexity in decoding with integrated language models (see Section 3), as well as synchronous parsing (Zhang et al. , 2006).
---------------------------------------------------
N07-1063:18	47:163	(Zhang et al. , 2006) binarize grammars into CNF normal form, while (Watanabe et al. , 2006) allow only Griebach-Normal form grammars.
---------------------------------------------------
N06-3004:19	33:68	We develop this intuition into a technique called synchronous binarization (Zhang et al. , 2006) which binarizes a synchronous production or treetranduction rule on both source and target sides simultaneously.
---------------------------------------------------
D07-1078:20	144:178	We used a bottom-up, CKY-style decoder that works with binary xRs rules obtained via a synchronous binarization procedure (Zhang et al. , 2006).
---------------------------------------------------
D07-1078:21	35:178	2 Related Research Several researchers (Melamed et al. , 2004; Zhang et al. , 2006) have already proposed methods for binarizing synchronous grammars in the context of machine translation.
---------------------------------------------------
W07-0403:22	126:234	We can use a linear-time algorithm (Zhang et al. , 2006) to detect non-ITG movement in our high-confidence links, and remove the offending sentence pairs from our training corpus.
---------------------------------------------------
P06-1123:23	39:221	Decomposing the translational equivalence relations in the training data into smaller units of knowledge can improve a models ability to generalize (Zhang et al. , 2006).
---------------------------------------------------
P06-1123:24	21:221	So unlike some other studies (Zens and Ney, 2003; Zhang et al. , 2006), we used manually annotated alignments instead of automatically generated ones.
---------------------------------------------------
D08-1060:25	115:222	A CYK-style decoder has to rely on binarization to preprocess the grammar as did in (Zhang et al., 2006) to handle multi-nonterminal rules.
---------------------------------------------------
D08-1060:26	43:222	Work in (Al-Onaizan and Kishore, 2006; Xiong et al., 2006; Zens et al., 2004; Kumar and Byrne, 2005; Tillmann and Zhang, 2005) modeled the limited information available at phrase-boundaries.
---------------------------------------------------
W06-1606:27	106:175	The decoder uses a binarized representation of the rules, which is obtained via a syncronous binarization procedure (Zhang et al. , 2006).
---------------------------------------------------
D09-1007:28	140:225	Binarizing the grammars (Zhang et al., 2006) further increases the size of these sets, due to the introduction of virtual nonterminals.
---------------------------------------------------
D09-1037:29	204:224	7Our decoder lacks certain features shown to be beneficial to synchronous grammar decoding, in particular rule binarisation (Zhang et al., 2006).
---------------------------------------------------
P09-2036:30	91:110	model reranking has also been established, both for synchronous binarization (Zhang et al., 2006) and for target-only binarization (Huang, 2007).
---------------------------------------------------
P09-2036:31	10:110	Past work has synchronously binarized such rules for efficiency (Zhang et al., 2006; Huang et al., 2008).
---------------------------------------------------
P06-1121:32	152:176	Its rule binarization is described in (Zhang et al. , 2006).
---------------------------------------------------
P08-1023:33	10:135	Compared with their string-based counterparts, treebased systems offer some attractive features: they are much faster in decoding (linear time vs. cubic time, see (Huang et al., 2006)), do not require a binary-branching grammar as in string-based models (Zhang et al., 2006), and can have separate grammars for parsing and translation, say, a context-free grammar for the former and a tree substitution grammar for the latter (Huang et al., 2006).
---------------------------------------------------
N09-1026:34	58:239	Rulesize and lexicalization affect parsing complexity whether the grammar is binarized explicitly (Zhang et al., 2006) or implicitly binarized using Early-style intermediate symbols (Zollmann et al., 2006).
---------------------------------------------------
W08-0403:35	17:207	Recent work by (Zhang et al., 2006) shows a practically ef cient approach that binarizes linguistically SCFG rules when possible.
---------------------------------------------------
N09-1049:36	32:210	Extensions to Hiero Several authors describe extensions to Hiero, to incorporate additional syntactic information (Zollmann and Venugopal, 2006; Zhang and Gildea, 2006; Shen et al., 2008; Marton and Resnik, 2008), or to combine it with discriminative latent models (Blunsom et al., 2008).
---------------------------------------------------
D08-1066:37	47:243	Our work expands on the general approach taken by (DeNero et al., 2006; Moore and Quirk, 2007) but arrives at insights similar to those of the most recent work (Zhang et al., 2006), albeit in a completely different manner.
---------------------------------------------------
D08-1066:38	32:243	This is in line with earlier work on consistent estimation for similar models (Zollmann and Simaan, 2006), and agrees with the most up-to-date work that employs Bayesian priors over the estimates (Zhang et al., 2008).
---------------------------------------------------
D08-1066:39	9:243	While this heuristic estimator gives good empirical results, it does not seem to optimize any intuitively reasonable objective function of the (wordaligned) parallel corpus (see e.g., (DeNero et al., 2006)) The mounting number of efforts attacking this problem over the last few years (DeNero et al., 2006; Marcu and Wong, 2002; Birch et al., 2006; Moore and Quirk, 2007; Zhang et al., 2008) exhibits its difficulty.
---------------------------------------------------
D08-1066:40	66:243	3.1 Binarizable segmentations (a) Following (Zhang et al., 2006; Huang et al., 2008), every sequence of phrase alignments can be viewed 1For example, if the cut-off on phrase pairs is ten words, all sentence pairs smaller than ten words in the training data will be included as phrase pairs as well.
---------------------------------------------------
D08-1066:41	78:243	(Zhang et al., 2006; Huang et al., 2008)), a binarizable segmentation/permutation can be recognized by a binarized Synchronous Context-Free Grammar (SCFG), i.e., an SCFG in which the right hand sides of all non-lexical rules constitute binarizable permutations.
---------------------------------------------------
W07-0412:42	135:166	On the positive side, recent work exploring the automaticbinarizationofsynchronousgrammars(Zhang et al. , 2006) has indicated that non-binarizable constructions seem to be relatively rare in practice.
---------------------------------------------------
D07-1079:43	48:289	Translation rules can:  look like phrase pairs with syntax decoration: NPB(NNP(prime) NNP(minister) NNP(keizo) NNP(obuchi)) BUFDFKEUBWAZ  carry extra contextual constraints: VP(VBD(said) x0:SBAR-C) DKx0 (according to this rule, DK can translate to said only if some Chinese sequence to the right ofDK is translated into an SBAR-C)  be non-constituent phrases: VP(VBD(said) SBAR-C(IN(that) x0:S-C)) DKx0 VP(VBD(pointed) PRT(RP(out)) x0:SBAR-C) DXGPx0  contain non-contiguous phrases, effectively phrases with holes: PP(IN(on) NP-C(NPB(DT(the) x0:NNP)) NN(issue)))) GRx0 EVABG6 PP(IN(on) NP-C(NPB(DT(the) NN(issue)) x0:PP)) GRx0 EVEVABABG6  be purely structural (no words): S(x0:NP-C x1:VP)x0 x1  re-order their children: NP-C(NPB(DT(the) x0:NN) PP(IN(of) x1:NP-C)) x1 DFx0 Decoding with this model produces a tree in the target language, bottom-up, by parsing the foreign string using a CYK parser and a binarized rule set (Zhang et al. , 2006).
---------------------------------------------------
