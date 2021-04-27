P09-2035:1	9:92	1 Introduction Tree-based statistical machine translation models in days have witness promising progress in recent years, such as tree-to-string models (Liu et al., 2006; Huang et al., 2006), tree-to-tree models (Quirk et al.,2005;Zhang et al., 2008).
---------------------------------------------------
P09-2035:2	13:92	It will inevitably hurt the translation quality (Quirk and Corston-Oliver, 2006; Mi and Huang, 2008).
---------------------------------------------------
P09-2035:3	79:92	Forest-based rule extractor (Mi and Huang 2008) is used with a pruning threshold p=3.
---------------------------------------------------
P09-1065:4	215:227	Hypergraphs have been successfully used in parsing (Klein and Manning., 2001; Huang and Chiang, 2005; Huang, 2008) and machine translation (Huang and Chiang, 2007; Mi et al., 2008; Mi and Huang, 2008).
---------------------------------------------------
D09-1127:5	16:211	For example, Smith and Smith (2004) and Burkett and Klein (2008) show that joint parsing (or reranking) on a bitext improves accuracies on either or both sides by leveraging bilingual constraints, which is very promising for syntax-based machine translation which requires (good-quality) parse trees for rule extraction (Galley et al., 2004; Mi and Huang, 2008).
---------------------------------------------------
P09-1020:6	100:277	4 Training This section discusses how to extract our translation rules given a triple nullnull,null null ,nullnull . As we know, the traditional tree-to-string rules can be easily extracted from nullnull,null null ,nullnull  using the algorithm of Mi and Huang (2008) 2 . We would like  2  Mi and Huang (2008) extend the tree-based rule extraction algorithm (Galley et al., 2004) to forest-based by introducing non-deterministic mechanism.
---------------------------------------------------
P09-1020:7	216:277	For parse forest pruning (Mi et al., 2008), we utilize the Margin-based pruning algorithm presented in (Huang, 2008).
---------------------------------------------------
P09-1020:8	150:277	Finally, to calculate rule feature probabilities for our model, we need to calculate the fractional counts (it is a kind of probability defined in Mi and Huang, 2008) of each translation rule in a parse forest.
---------------------------------------------------
P09-1020:9	151:277	In the tree case, we can use the inside-outside-based methods (Mi and Huang 2008) to do it.
---------------------------------------------------
P09-1020:10	45:277	Mi and Huang (2008) propose a forest-based rule extraction algorithm, which learn tree to string rules from source forest and target string.
---------------------------------------------------
P09-1020:11	11:277	Among these advances, forest-based modeling (Mi et al., 2008; Mi and Huang, 2008) and tree sequence-based modeling (Liu et al., 2007; Zhang et al., 2008a) are two interesting modeling methods with promising results reported.
---------------------------------------------------
P09-1020:12	107:277	2) Employ the forest-based tree rule extraction algorithm (Mi and Huang, 2008) to extract our rules from the non-complete forest.
---------------------------------------------------
P09-1020:13	149:277	Then we can easily extract our rules from the CF using the tree rule extraction algorithm (Mi and Huang, 2008).
---------------------------------------------------
P09-1089:14	229:234	Finally, some researchers have advocated recently the use of shared structures such as parse forests (Mi and Huang, 2008) or word lattices (Dyer et al., 2008) in order to allow a compact representation of alternative inputs to an SMT system.
---------------------------------------------------
D09-1073:15	7:210	1 Introduction Phrase-based method (Koehn et al., 2003; Och and Ney, 2004; Koehn et al., 2007) and syntaxbased method (Wu, 1997; Yamada and Knight, 2001; Eisner, 2003; Chiang, 2005; Cowan et al., 2006; Marcu et al., 2006; Liu et al., 2007; Zhang et al., 2007c, 2008a, 2008b; Shen et al., 2008; Mi and Huang, 2008) represent the state-of-the-art technologies in statistical machine translation (SMT).
---------------------------------------------------
