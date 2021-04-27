W06-1518:1	83:103	(Gildea and Jurafsky, 2002; Punyakanok, Roth and Yih, 2005; Pradhan et al, 2005; Toutanova et al. , 2005)), the focus has been on two different aspects of the SRL task: (a) finding appropriate features, and (b) resolving the parsing accuracy problem by combining multiple parsers/predictions.
---------------------------------------------------
P08-1040:2	201:218	Approaches include incorporating a subcategorization feature (Gildea & Jurafsky, 2002; Xue & Palmer, 2004), such as the one used in our baseline; and building a model which jointly classifies all arguments of a verb (Toutanova et al., 2005).
---------------------------------------------------
P06-2034:3	15:160	While reranking has benefited many tagging and parsing tasks (Collins, 2000; Collins, 2002c; Charniak and Johnson, 2005) including semantic role labeling (Toutanova et al. , 2005), it has not yet been applied to semantic parsing.
---------------------------------------------------
W06-2909:4	57:206	The usual approach (Toutanova et al. , 2005) uses a traditional boundary classifier (TBC) to select the set of potential argument nodes.
---------------------------------------------------
W06-2909:5	95:206	3.3 Re-ranking NSTs with Tree Kernels To implement the re-ranking model, we follow the approach described in (Toutanova et al. , 2005).
---------------------------------------------------
W06-2909:6	198:206	In (Toutanova et al. , 2005), it was observed that there are strong dependencies among the labels of the semantic argument nodes of a verb.
---------------------------------------------------
W06-2909:7	23:206	Moreover, we modeled SRL as a re-ranking task in line with (Toutanova et al. , 2005).
---------------------------------------------------
W06-2909:8	100:206	We adopt the same algorithm described in (Toutanova et al. , 2005).
---------------------------------------------------
W06-2909:9	81:206	The re-ranking approach is the most promising one as suggested in (Toutanova et al. , 2005) but it does not clearly reveal if tree kernels can be used to learn the difference between correct or incorrect argument structures.
---------------------------------------------------
W06-2909:10	159:206	66 (Punyakanok et al. , 2005; Toutanova et al. , 2005; Pradhan et al. , 2005b) were obtained by exploiting the information on the whole predicate argument structure.
---------------------------------------------------
D08-1008:11	15:220	Following the seminal work of Gildea and Jurafsky (2002), there have been many extensions in machine learning models, feature engineering (Xue and Palmer, 2004), and inference procedures (Toutanova et al., 2005; Surdeanu et al., 2007; Punyakanok et al., 2008).
---------------------------------------------------
N06-1055:12	9:170	It is generally formulated as a semantic role labeling (SRL) task, where each argument of the predicate is assigned a label that represents the semantic role it plays with regard to its predicate (Gildea and Jurafsky, 2002; Hacioglu et al. , 2003; Pradhan et al. , 2004b; Xue and Palmer, 2004; Toutanova et al. , 2005; Koomen et al. , 2005).
---------------------------------------------------
N06-1055:13	139:170	4.5 Reranking In a recent paper on the SRL on verbal predicates for English, (Toutanova et al. , 2005) pointed out that one potential flaw in a SRL system where each argument is considered on its own is that it does not take advantage of the fact that the arguments (not the adjuncts) of a predicate are subject to the hard constraint that they do not have the same label3.
---------------------------------------------------
J08-2001:14	47:86	Also in this issue, Toutanova, Haghighi, and Manning apply re-ranking to select the best among a set of candidate complete solutions produced by a base SRL system.Finally, probabilistic models have also been applied to produce the structured output, for example, generative models (Thompson, Levy, and Manning 2003), sequence tagging with classiers (M`arquez et al. 2005; Pradhan et al.2005b), and Conditional Random Fields on tree structures (Cohn and Blunsom 2005).These approaches at a global level may demand considerable extra computation, but current optimization techniques help solve them quite efciently.
---------------------------------------------------
J08-2001:15	56:86	Joint scoring and combination components open the door to richer types of features, which may take into account global properties of the candidate solution plus dependencies among the different arguments.The most remarkable work in this direction is the reranking approach by Toutanova, Haghighi, and Manning in this issue.When training the ranker to select the best candidate solution they codify pattern features as strings containing the whole argument structure of the candidate.Several variations of this type of feature (with different degrees of generalization to avoid sparseness) allow them to signicantly increase the performance of the base system.Also related, Pradhan et al.(2005b) and Surdeanu et al.(2007) convert the condence scores of several base SRL systems into features for training a nal machine learningbased combination system.Surdeanu et al.(2007) develop a broad spectrum of features, with sentencebased information, describing the role played by the candidate argument in every solution proposed by the different base SRL systems.
---------------------------------------------------
J08-2001:16	50:86	An important consideration within this general SRL architecture is the combination of systems and input annotations.Most SRL systems include some kind of combination to increase robustness, gain coverage, and reduce effects of parse errors.One may combine: (1) the output of several independent SRL basic systems (Surdeanu et al.2007; Pradhan et al.2005b), or (2) several outputs from the same SRL system obtained by changing input annotations or other internal parameters (Koomen et al. 2005; Toutanova, Haghighi, and Manning 2005).The combination can be as simple as selecting the best among the set of complete candidate solutions, but usually consists of combining fragments of alternative solutions to construct the nal output.Finally, the combination component may involve machine learning or not.The gain in performance from the combination step is consistently between two and three F 1 points.However, a combination approach increases system complexity and penalizes efciency.
---------------------------------------------------
D09-1059:17	15:170	The top-performing system in the task (Johansson and Nugues, 2008) applied a very simple reranking scheme by means of a k-best syntactic output, similar to previous attempts (Gildea and Jurafsky, 2002; Toutanova et al., 2005) to improve semantic role labeling performance by using mul561 tiple parses.
---------------------------------------------------
C08-1050:18	64:214	Although state-of-the-art SRL systems use sophisticated statistical models to perform these two tasks jointly (e.g. Toutanova et al., 2005, Johansson and Nugues, 2008), we implemented them as two independent support vector classifiers to be able to analyze the impact of syntactic representation on each task separately.
---------------------------------------------------
C08-1050:19	30:214	However, except from a few tentative experiments (Toutanova et al., 2005), grammatical function is not explicitly used by current automatic SRL systems, but instead emulated from constituent trees by features like the constituent position and the governing category.
---------------------------------------------------
W06-1617:20	96:156	4.2.4 Neighboring arguments The research of (Jiang et al. , 2005; Toutanova et al. , 2005) has shown the importance of capturing information of the global argument frame in order to correctly classify the local argument.
---------------------------------------------------
W06-1617:21	45:156	During testing, the algorithm of enforcing nonoverlapping arguments by (Toutanova et al. , 2005) is used.
---------------------------------------------------
W06-1668:22	147:239	The problem is, given that a node has a core argument label, decide what the correct label is. Other researchers have also looked at this subproblem (Gildea and Jurafsky, 2002; Toutanova et al. , 2005; Pradhan et al. , 2005a; Xue and Palmer, 2004).
---------------------------------------------------
W06-1668:23	202:239	To put our results in the context of previous work, other results on core arguments using the same input features have been reported, the best being 91.4% for an SVM with a degree 2 polynomial kernel (Pradhan et al. , 2005a).3 The highest reported result for independent classification of core arguments is 96.0% for a log-linear model using more than 20 additional basic features (Toutanova et al. , 2005).
---------------------------------------------------
W06-1668:24	8:239	These include models for part-of-speech tagging (Toutanova et al. , 2003), semantic-role labeling (Punyakanok et al. , 2005; Pradhan et al. , 2005b) and Penn Treebank parsing (Charniak and Johnson, 2005).
---------------------------------------------------
W07-2048:25	88:92	In addition, while the system described here is based on pipelined classification, recent research on semantic role labeling has shown that significant performance improvements can be gained by exploiting interdependencies between arguments (Toutanova et al. , 2005).
---------------------------------------------------
W05-0623:26	44:79	Most of the features we use are described in more detail in (Toutanova et al. , 2005).
---------------------------------------------------
W05-0623:27	41:79	We find the exact top N consistent1 most likely local model labelings using a simple dynamic program described in (Toutanova et al. , 2005).
---------------------------------------------------
W05-0623:28	24:79	The ones denoted with asterisks (*) were not present in (Toutanova et al. , 2005).
---------------------------------------------------
W05-0623:29	3:79	The system, introduced in (Toutanova et al. , 2005), implements a joint model that captures dependencies among arguments of a predicate using log-linear models in a discriminative re-ranking framework.
---------------------------------------------------
W05-0623:30	77:79	The improvement achieved by the joint model relative to the local model is about 2 points absolute in F-Measure, similar to the improvement when gold-standard syntactic parses are used (Toutanova et al. , 2005).
---------------------------------------------------
D09-1004:31	76:221	The concept of support verb was broadly used (Toutanova et al., 2005; Xue, 2006; Jiang and Ng, 2006)4, we here extend it to nouns and prepositions.
---------------------------------------------------
D09-1004:32	68:221	These works include (Gildea and Jurafsky, 2002; Carreras and Marquez, 2005; Koomen et al., 2005; Marquez et al., 2005; Dang and Palmer, 2005; Pradhan et al., 2005; Toutanova et al., 2005; Jiang and Ng, 2006; Liu and Ng, 2007; Surdeanu et al., 2007; Johansson and Nugues, 2008; Che et al., 2008).
---------------------------------------------------
D09-1004:33	88:221	4Note that the meaning of support verb is slightly different between (Toutanova et al., 2005) and (Xue, 2006; Jiang and Ng, 2006) 32 first includes all syntactic children (children), the second also includes all but excludes the left most and the right most children (noFarChildren).
---------------------------------------------------
H05-1049:34	65:196	Semantic Role Labeling: We also augment the graph representation with Probank-style semantic roles via the system described in (Toutanova et al. , 2005).
---------------------------------------------------
W06-2607:35	202:203	Finally, as CoNLL 2005 has shown that the most important contribution relates on re-ranking predicate argument structures based on one single tree (Toutanova et al. , 2005) or several trees (Punyakanok et al. , 2005), we would like to use tree kernels for the re-ranking task.
---------------------------------------------------
W06-2607:36	88:203	3.2 Kernels on complete predicate argument structures The type of a target argument strongly depends on the type and number of the predicates arguments1 (Punyakanok et al. , 2005; Toutanova et al. , 2005).
---------------------------------------------------
P07-1036:37	127:270	We note that in the presence of constraints, the inference procedure (for nding the output y that maximizes the cost function) is usually done with search techniques (rather than Viterbi decoding, see (Toutanova et al. , 2005; Roth and Yih, 2005) for a discussion), we chose beamsearch decoding.
---------------------------------------------------
P07-1036:38	12:270	On the other hand, in the supervised setting, it has been shown that incorporating domain and problem speci c structured information can result in substantial improvements (Toutanova et al. , 2005; Roth and Yih, 2005).
---------------------------------------------------
P07-1036:39	38:270	(Punyakanok et al. , 2005; Toutanova et al. , 2005; Roth and Yih, 2005).
---------------------------------------------------
W09-1103:40	11:262	Statistical parsers are major components in NLP applications such as QA (Kwok et al., 2001), MT (Marcu et al., 2006) and SRL (Toutanova et al., 2005).
---------------------------------------------------
D07-1062:41	131:201	This result is better than (Xue and Palmer, 2004), and better on gold parses compared to (Toutanova et al. , 2005; Punyakanok et al. , 2005b).
---------------------------------------------------
D07-1062:42	11:201	For SRL, high accuracy has been achieved by: (i) proposing new types of features (see Table 1 in Section 3 for previously proposed features), (ii)modelingthepredicateframesetbycapturingdependencies between arguments (Gildea and Jurafsky, 2002; Pradhan et al. , 2004; Toutanova et al. , 2005; Punyakanok et al. , 2005a), (iii) dealing with incorrect parser output by using more than one parser (Pradhan et al. , 2005b).
---------------------------------------------------
D07-1062:43	120:201	Our evaluation criteria which is based on predictingtheSRLforconstituentsintheparsetreeisbased on the evaluation used in (Toutanova et al. , 2005).
---------------------------------------------------
W09-1206:44	21:126	Aside from the lack of a predicate identification module, which was not needed, as predicates were given, this architecture is identical to the one adopted by recent systems (Surdeanu et al., 2008), as well as the general approach within the field (Gildea and Jurafsky, 2002; Toutanova et al., 2005).
---------------------------------------------------
W07-0208:45	4:151	Rich annotations of corpora has allowed for the development of techniques for recovering deep linguistic structures: syntactic non-local dependencies (Johnson, 2002; Hockenmaier, 2003; Dienes, 2004; Jijkoun and de Rijke, 2004) and semantic arguments (Gildea, 2001; Pradhan et al. , 2005; Toutanovaetal.
---------------------------------------------------
W06-2908:46	9:215	As this task is recognized as an important step after (or the last step of) syntactic analysis, many studies have been conducted to achieve accurate semantic role labeling (Gildea and Jurafsky, 2002; Moschitti, 2004; Hacioglu et al. , 2004; Punyakanok et al. , 2004; Pradhan et al. , 2005a; Pradhan et al. , 2005b; Toutanova et al. , 2005).
---------------------------------------------------
W06-1673:47	15:221	(2005)), but efficiently enumerating k-best lists often requires very substantial cognitive and engineering effort, e.g., in (Huang and Chiang, 2005; Toutanova et al. , 2005).
---------------------------------------------------
W06-1673:48	127:221	(2005) and Toutanova et al.
---------------------------------------------------
N07-1069:49	16:158	Many researchers have investigated applying machine learning to corpus specifically annotated with this task in mind, PropBank, since 2000 (Chen and Rambow, 2003; Gildea and Hockenmaier, 2003; Hacioglu et al. , 2003; Moschitti, 2004; Yi and Palmer, 2004; Pradhan et al. , 2005b; Punyakanok et al. , 2005; Toutanova et al. , 2005).
---------------------------------------------------
W09-2211:50	5:28	Perhaps more importantly, discriminative models have been shown to offer competitive performance on a variety of sequential and structured learning tasks in NLP that are traditionally tackled via generative models , such as letter-to-phoneme conversion (Jiampojamarn et al., 2008), semantic role labeling (Toutanova et al., 2005), syntactic parsing (Taskar et al., 2004), language modeling (Roark et al., 2004), and machine translation (Liang et al., 2006).
---------------------------------------------------
W08-2133:51	106:121	(2005) and Toutanova et al.
---------------------------------------------------
P09-1054:52	9:190	The applications range from simple classification tasks such as text classification and history-based tagging (Ratnaparkhi, 1996) to more complex structured prediction tasks such as partof-speech (POS) tagging (Lafferty et al., 2001), syntactic parsing (Clark and Curran, 2004) and semantic role labeling (Toutanova et al., 2005).
---------------------------------------------------
P07-1027:53	85:194	Accordingly, we do not maximize the probability of the entire labeled parse tree as in (Toutanova et al. , 2005).
---------------------------------------------------
C08-1105:54	15:211	Pradhan et al., 2005), analyzing the complex input  syntax trees (Moschitti, 2004; Liu and Sarkar, 2007), exploiting the complicated output  the predicate-structure (Toutanova et al., 2005), as well as capturing paradigmatic relations between predicates (Gordon and Swanson, 2007).
---------------------------------------------------
