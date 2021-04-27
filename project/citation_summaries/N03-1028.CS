W06-0113:1	204:224	6 Related works After the work of Ramshaw and Marcus (1995), many machine learning techniques have been applied to the basic chunking task, such as Support Vector Machines (Kudo and Matsumoto, 2001), Hidden Markov Model(Molina and Pla 2002), Memory Based Learning (Sang, 2002), Conditional Random Fields (Sha and Pereira, 2003), and so on.
---------------------------------------------------
D09-1119:2	45:258	As discussed above, all state-of-the-art published methods rely on lexical features for such tasks (Zhang et al., 2001; Sha and Pereira, 2003; Finkel et al., 2005; Ratinov and Roth, 2009).
---------------------------------------------------
P06-1028:3	182:242	5.1 Comparison Methods and Parameters For ML and MAP, we performed exactly the same training procedure described in (Sha and Pereira, 2003) with L-BFGS optimization.
---------------------------------------------------
P06-1028:4	175:242	3 with a variant of the forwardbackward algorithm (Sha and Pereira, 2003).
---------------------------------------------------
P06-1028:5	28:242	Z(x)=summationtext yY producttext cC(y,x) c(y,x; ) is a normalization factor over all output values, Y. Following the definitions of (Sha and Pereira, 2003), a log-linear combination of weighted features, c(y,x; ) = exp( fc(y,x)), is used as individual potential functions, where fc represents a feature vector obtained from the corresponding clique c. That is, producttextcC(y,x) c(y,x) = exp( F(y,x)), where F(y,x)=summationtextc fc(y,x) is the CRFs global feature vector for x and y. The most probable output y is given by y = arg maxyY p(yjx; ).
---------------------------------------------------
P06-1028:6	42:242	The details of actual optimization procedures for linear chain CRFs, which are typical CRF applications, have already been reported (Sha and Pereira, 2003).
---------------------------------------------------
P06-1028:7	104:242	12, by using the variant of the forward-backward and Viterbi algorithm described in (Sha and Pereira, 2003).
---------------------------------------------------
P06-1028:8	12:242	The maximum a posteriori (MAP) criterion over parameters, , given x and y is the natural choice for reducing over-fitting (Sha and Pereira, 2003).
---------------------------------------------------
P06-1028:9	8:242	The modeling power of CRFs has been of great benefit in several applications, such as shallow parsing (Sha and Pereira, 2003) and information extraction (McCallum and Li, 2003).
---------------------------------------------------
E09-1090:10	79:207	The difference between our CRF chunker and that in (Sha and Pereira, 2003) is that we could not use second-order CRF models, hence we could not use trigram features on the BIO states.
---------------------------------------------------
E09-1090:11	77:207	Since the task is basically identical to shallow parsing by CRFs, we follow the feature sets used in the previous work by Sha and Pereira (2003).
---------------------------------------------------
E09-1090:12	147:207	Although not directly comparable, Sha and Pereira (2003) report almost the same level of accuracy (94.38%) on noun phrase recognition, using a much smaller training set.
---------------------------------------------------
J07-4004:13	40:948	Initially we used generalized iterative scaling (GIS) for the parsing models described here, but found that convergence was extremely slow; Sha and Pereira (2003) present a similar finding for globally optimized log-linear models for sequences.
---------------------------------------------------
I08-1044:14	37:215	CRFs have been successfully applied to a number of real-world tasks, including NP chunking (Sha and Pereira, 2003), Chinese word segmentation (Peng et al., 2004), information extraction (Pinto et al., 2003; Peng and McCallum, 2004), named entity identification (McCallum and Li, 2003; Settles, 2004), and many others.
---------------------------------------------------
P04-1007:15	71:275	2.3 Conditional Random Fields Conditional Random Fields have been applied to NLP tasks such as parsing (Ratnaparkhi et al. , 1994; Johnson et al. , 1999), and tagging or segmentation tasks (Lafferty et al. , 2001; Sha and Pereira, 2003; McCallum and Li, 2003; Pinto et al. , 2003).
---------------------------------------------------
P06-3006:16	111:138	Algorithm 4.1: DECODE PATH(x,n,start,go) for each label y1 node[0][y1].costs(y1,0) node[0][y1].endstart; for j1 to n1 for each label yj+1 bestc; endnull; for each label yj costnode[j][yj].cost +s(yj,yj+1,j,j + 1) +s(yj+1,j + 1); endnode[j][yj].end; if (yj negationslash= yj+1) endgo(yj+1,end); if (cost > bestcendnegationslash= null) bestccost; if (bestcnegationslash=) node[j + 1][yj+1].costbestc; node[j + 1][yj+1].endend; bestc; endnull; for each label yn if (node[j][yn].cost > bestc) bestcnode[j][yn].cost; endnode[j][yn].end; return (bestc,end) 34 4.1.3 Learning: Conditional Random Fields Given the above problem formulation, we trained the linear-chain undirected graphical model as Conditional Random Fields (Lafferty et al, 2001; Sha and Pereira, 2003), one of the best performing chunkers.
---------------------------------------------------
P06-3006:17	113:138	A detailed description of CRFs can be found in (Lafferty et al, 2001; Sha and Pereira, 2003; Malouf, 2002; Peng and McCallum, 2004).
---------------------------------------------------
W06-1615:18	9:260	Discriminative taggers and chunkers have been the state-of-the-art for more than a decade (Ratnaparkhi, 1996; Sha and Pereira, 2003).
---------------------------------------------------
W06-1615:19	253:260	We are also focusing on other potential applications, including chunking (Sha and Pereira, 2003), named entity recognition (Florian et al. , 2004; Ando and Zhang, 2005b; Daume III and Marcu, 2006), and speaker adaptation (Kuhn et al. , 1998).
---------------------------------------------------
W06-1643:20	91:186	Specifically, to account for skip-edges, we used a technique inspired by (Sha and Pereira, 2003), in which multiple state dependencies, such as an order-2 Markov model, are encoded using auxiliary tags.
---------------------------------------------------
W06-0112:21	57:131	We define a transition matrix as following: '' (, | ) exp( (,,, )) ijj j M yyx f yyxi=  Then, 1 1 1 1 (|,) (, |) () n ii i i py x M y y x Zx  +  = =  and let * denote component-wise matrix product, (|,) 1 (, ) ( |,) (, ) () () () 1 k pYx k k k y T ii ii i T n EFYx pYyxFyx fM Zx Zx a     ==  = =   Where ii  , as the forward and backward state-cost vectors defined by 1 11 1, 101 T ii T ii Min M in i in     ++ 0<   <  = =  = =   Sha & Pereira (2003) provided a thorough discussion of CRF training methods including preconditioned Conjugate Gradient, limitedMemory Quasi-Newton and voted perceptron.
---------------------------------------------------
W06-0112:22	19:131	(Fei and Fernando, 2003) considered chunking as a sequence labeling task and achieved good performance by an improved training methods of CRF.
---------------------------------------------------
W06-0112:23	53:131	an 2.3 Conditional Random Fields Lafferty et al.( 2001) present the Conditional Random Fields for building probabilistic models to segment and label sequence data, which was used effectively for base NP chunking (Sha & Pereira, 2003).
---------------------------------------------------
W06-0112:24	60:131	The results of CRF are better than that of SVM, which is the same as the outcome of the English base NP chunking in (Sha & Pereira, 2003).
---------------------------------------------------
H05-1094:25	46:186	For more information on current training methods for CRFs, see Sha and Pereira (2003).
---------------------------------------------------
H05-1040:26	40:213	Using a parse of the question sentence, we derive a novel set of multi-resolution features suitable for training a conditional random field (CRF) (Lafferty et al. , 2001; Sha and Pereira, 2003).
---------------------------------------------------
H05-1040:27	122:213	Hidden Markov models are extremely popular for such applications, but recent work has shown that conditional random fields (CRFs) (Lafferty et al. , 2001; Sha and Pereira, 2003) have a consistent advantage over traditional HMMs in the face of many redundant features.
---------------------------------------------------
W06-1670:28	121:362	1: Initialize w0 =vector0 2: for t = 1,T do 3: Choose xi 4: Compute y = argmaxyY F(xi,y;w) 5: if yinegationslash= y then 6: wt+1wt + (xi,yi)(xi, y) 7: end if 8: w = 1T summationtexttwt 9: end for 10: return w the perceptron performance is comparable to that of Conditional Random Field models (Sha and Pereira, 2003), The tendency to overfit of the perceptron can be mitigated in a number of ways including regularization and voting.
---------------------------------------------------
P05-2004:29	90:237	See, for example, NP chunkers utilizing conditional random fields (Sha and Pereira, 2003) and support vector machines (Kudo and Matsumoto, 2001).
---------------------------------------------------
C04-1081:30	54:178	However, our implementation uses a quasi-Newton gradient-climber BFGS for optimization, which has been shown to converge much faster (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
C08-1106:31	12:203	extensive comparisons among methods (McDonald 2005; Sha & Pereira 2003; Kudo & Matsumoto 2001).
---------------------------------------------------
C08-1106:32	121:203	We employ similar predicate sets defined in Sha & Pereira (2003).
---------------------------------------------------
C08-1106:33	159:203	Nevertheless, since testing the significance of shallow parsers F-measures is tricky, individual labeling accuracy provides a more convenient basis for statistical significance tests (Sha & Pereira 2003).
---------------------------------------------------
C08-1106:34	138:203	Since the CRF model is one of the successful models in sequential labeling tasks (Lafferty et al. 2001; Sha & Pereira 2003; McDonald et al. 2005), in this section, we also compare LDCRFs with CRFs.
---------------------------------------------------
C08-1106:35	48:203	As for the task of shallow parsing, CRFs also outperform many other state-of-the-art models (Sha & Pereira 2003; McDonald et al. 2005).
---------------------------------------------------
C08-1106:36	183:203	We observe that the L-BFGS optimizer is slightly faster than CG on LDCRFs (see Figure 3), which echoes the comparison between the L-BFGS and the CG optimizing technique on the CRF model (Sha & Pereira 2003).
---------------------------------------------------
P05-1002:37	13:188	CRFs have been applied with impressive empirical results to the tasks of named entity recognition (McCallum and Li, 2003), simplified part-of-speech (POS) tagging (Lafferty et al. , 2001), noun phrase chunking (Sha and Pereira, 2003) and extraction of tabular data (Pinto et al. , 2003), among other tasks.
---------------------------------------------------
P05-1001:38	212:242	Comparison with previous best results: KM01 (Kudoh and Matsumoto, 2001), CM03 (Carreras and Marquez, 2003), SP03 (Sha and Pereira, 2003), ZDJ02 (Zhang et al. , 2002).
---------------------------------------------------
P06-1141:39	34:181	2 Conditional Random Fields We use a Conditional Random Field (Lafferty et al. , 2001; Sha and Pereira, 2003) since it represents the state of the art in sequence modeling and has also been very effective at Named Entity Recognition.
---------------------------------------------------
W05-0622:40	12:69	CRFs have been applied with impressive empirical results to the tasks of named entity recognition (McCallum and Li, 2003; Cohn et al. , 2005), part-of-speech (PoS) tagging (Lafferty et al. , 2001), noun phrase chunking (Sha and Pereira, 2003) and extraction of table data (Pinto et al. , 2003), among other tasks.
---------------------------------------------------
W06-0505:41	54:181	Furthermore, CRFs have been used successfully in information extraction (Peng and McCallum, 2004), named entity recognition (Li and McCallum, 2003; McCallum and Li, 2003) and sentence parsing (Sha and Pereira, 2003).
---------------------------------------------------
W06-1313:42	35:168	An Interpreter component tags the ASR output with its its dialogue move and parameter labels using two separate Conditional Random Field (Sha and Pereira, 2003; McCallum, 2002) taggers trained on hand-annotated utterances.
---------------------------------------------------
W05-0402:43	62:232	CRFs have been shown to perform well in a number of natural language processing applications, such as POS tagging (Lafferty et al. , 2001), shallow parsing or NP chunking (Sha and Pereira, 2003), and named entity recognition (McCallum and Li, 2003).
---------------------------------------------------
P07-1080:44	155:171	tional Random Fields, are the standard tools for shallow parsing (Sha and Pereira, 2003).
---------------------------------------------------
W04-3111:45	146:153	While it is difficult to compare taggers that tag different types of entities (e.g. , (Friedman et al. , 2001; Gaizauskas et al. , 2003)), CRFs have been utilized for state-of-the-art results in NP-chunking and gene and protein tagging (Sha and Pereira, 2003; McDonald and Pereira, 2004) Currently, we are beginning to investigate methods to identify relations over the variation components that are extracted using the entity tagger.
---------------------------------------------------
P04-1086:46	36:181	Among these methods, CRFs is the most common technique used in NLP and has been successfully applied to Part-of-Speech Tagging (Lafferty et al. , 2001), Named-Entity Recognition (Collins, 2002) and shallow parsing (Sha and Pereira, 2003; McCallum, 2003).
---------------------------------------------------
P04-1086:47	60:181	However, gradient-based methods have often found to be more efficient for minimizing Equation 3 (Minka, 2001; Sha and Pereira, 2003).
---------------------------------------------------
W07-2217:48	79:259	Thechoiceismotivatedbythesimplicity and performance of perceptrons, which have proved competitive on a number of tasks; e.g., in shallow parsing, where perceptrons performance is comparable to that of Conditional Random Field models (Sha & Pereira, 2003).
---------------------------------------------------
W03-0430:49	7:87	CRFs have shown empirical successes recently in POS tagging (Lafferty et al. , 2001), noun phrase segmentation (Sha and Pereira, 2003) and Chinese word segmentation (McCallum and Feng, 2003).
---------------------------------------------------
W03-0430:50	46:87	There can easily be over 100,000 atomic tests (mostly based on tests for the identity of words in the vocabulary), and ten or more shifted-conjunction patternsresulting in several million features (Sha and Pereira, 2003).
---------------------------------------------------
W03-0430:51	41:87	It has recently been shown that quasi-Newton methods, such as L-BFGS, are significantly more efficient than traditional iterative scaling and even conjugate gradient (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
P08-1076:52	94:228	As regards the TIP1The second-order encoding used in our NER experiments is the same as that described in (Sha and Pereira, 2003) except removing IOB-tag of previous position label.
---------------------------------------------------
P08-1076:53	54:228	Moreover, we can obtain the same form of gradient as that of supervised CRFs (Sha and Pereira, 2003), that is, L1(prime|) =EP(Y,X;prime,)bracketleftbigh(Y,X)bracketrightbig  summationdisplay n EP(Y|xn;prime,)bracketleftbigh(Y,xn)bracketrightbig+logp(prime).
---------------------------------------------------
P06-2054:54	51:262	The L-BFGS method converges super-linearly to the solution, so it can be an efficient optimization technique on large-scale NLP problems (Sha and Pereira, 2003).
---------------------------------------------------
W06-0116:55	19:101	The model has been used for chunking(Sha and Pereira, 2003).
---------------------------------------------------
E09-1030:56	77:157	CRFs have been widely applied to tasks in natural language processing, especially those involving tagging words with labels such as partof-speech tagging and shallow parsing (Sha and Pereira, 2003), as well as sentence boundary detection (Liu et al., 2005; Liu et al., 2004).
---------------------------------------------------
E09-1088:57	110:222	Since the CRF model is one of the most successful models in sequential labeling tasks (Lafferty et al., 2001; Sha and Pereira, 2003), in this paper, we choosed CRFs as a baseline model for the comparison.
---------------------------------------------------
W04-1221:58	13:85	particular have been shown to be useful in partof-speech tagging (Lafferty et al. , 2001), shallow parsing (Sha and Pereira, 2003), and named entity recognition for newswire data (McCallum and Li, 2003).
---------------------------------------------------
I08-2078:59	79:119	In particular, the CRF trained by maximumlikelihood uses the L-BFGS algorithm (Liu and Nocedal, 1989), which converges quickly and gives a good performance on maximum entropy models (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
W06-1655:60	143:152	4 Relation to Previous Work There is a significant volume of work exploring the use of CRFs for a variety of chunking tasks, including named-entity recognition, gene prediction, shallow parsing and others (Finkel et al. , 2005; Culotta et al. , 2005; Sha and Pereira, 2003).
---------------------------------------------------
C08-1113:61	85:177	Although this non-concavity prevents efficient global maximization of equation (3), it still allows us to incorporate incomplete annotations using gradient ascent iterations (Sha and Pereira, 2003).
---------------------------------------------------
N06-1012:62	143:293	On the chunking task, the bagged model also outperforms the models of Kudo and Matsumoto (2001) and Sha and Pereira (2003), and equals the currently-best results of (Ando and Zhang, 2005), who use a large amount of unlabeled data.
---------------------------------------------------
N06-1012:63	131:293	The first bag also includes part-of-speech tags generated by the Brill tagger and the conjunctions of those tags used by Sha and Pereira (2003).
---------------------------------------------------
N06-1012:64	170:293	94.34 (Sha and Pereira, 2003) 94.38 (Kudo and Matsumoto, 2001) 94.39 (Ando and Zhang, 2005) 94.70 Combined CRF 94.77 Table 3: Results for the CoNLL 2000 Chunking Task.
---------------------------------------------------
N06-1012:65	10:293	1 Introduction Discriminative methods for training probabilistic models have enjoyed wide popularity in natural language processing, such as in part-of-speech tagging (Toutanova et al. , 2003), chunking (Sha and Pereira, 2003), namedentity recognition (Florian et al. , 2003; Chieu and Ng, 2003), and most recently parsing (Taskar et al. , 2004).
---------------------------------------------------
W08-2118:66	82:194	The conditional random fields (CRF) (Lafferty et al., 2001) model has shown great benefits in similar applications of natural language processing such as part-of-speech tagging, noun phrase chunking (Sha and Pereira, 2003), morphology disambiguation(Smith et al., 2005).
---------------------------------------------------
D07-1084:67	15:269	(Sha andPereira, 2003; SuttonandMcCallum, 2006; McCallum et al. , 2000; Alpaydin, 2004) Each of these Sequential Viterbi Models defines a set of scoring functions that evaluate fixed-size pieces of the output sequence based on fixed-size pieces of the input sequence.1 The overall score for 1For HMMs and MEMMs, the local scores are negative log probabilities.
---------------------------------------------------
W08-0301:68	82:195	Inspired by research work like (Lafferty et al., 2001) and (Sha and Pereira, 2003), our SWD model 3 uses first-order Conditional Random Field (CRF) to tackle the tagging task.1 The CRF model uses the following feature templates: 1.
---------------------------------------------------
I08-5013:69	18:193	They have shown to be useful in part of speech tagging (Lafferty et al. 2001), shallow parsing (Sha and Pereira 2003), and named entity recognition for Hindi newswire data (Li and McCallum 2003).
---------------------------------------------------
N04-1042:70	22:128	CRFs have been previously applied to other tasks such as name entity extraction (McCallum and Li, 2003), table extraction (Pinto et al. , 2003) and shallow parsing (Sha and Pereira, 2003).
---------------------------------------------------
N04-1042:71	40:128	Traditional maximum entropy learning algorithms, such as GIS and IIS (Pietra et al. , 1995), can be used to train CRFs, however, it has been found that a quasi-Newton gradient-climber, BFGS, converges much faster (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
W05-0611:72	168:171	Additionally, a broader comparison with point-wise predictors (Kashima and Tsuboi, 2004) as well as Viterbi-based probabilistic models (McCallum et al. , 2000; Lafferty et al. , 2001; Sha and Pereira, 2003) in large-scale comparative studies is warranted.
---------------------------------------------------
H05-1060:73	87:281	When training a CRF for POS tagging, IOB chunking (Sha and Pereira, 2003), or word segmentation (Peng et al. , 2004), one typically structures the conditional probabilities (in the objective function) using domain knowledge: in POS tagging, the set of allowed tags for a word is used; in IOB chunking, the bigram O I is disallowed; and in segmentation, a lexicon is used to enumerate the possible word boundaries.4 4This refinement is in the same vein as the move from maximum likelihood estimation to conditional estimation.
---------------------------------------------------
P09-2071:74	92:131	We used complete and dense input/output joint features for dense model (Dense), and only supported features that are used at least once in the training examples for sparse form better, the sparse model performs well in practice without significant loss of accuracy (Sha and Pereira, 2003).
---------------------------------------------------
P05-1024:75	144:200	As a baseline model, we used a shallow parser based on Conditional Random Fields (CRFs), very similar to that described in (Sha and Pereira, 2003).
---------------------------------------------------
W05-0707:76	122:239	CRFs have been shown to perform well in a number of natural language processing applications, such as POS tagging (Lafferty et al. , 2001), shallow parsing or NP chunking (Sha and Pereira, 2003), and named entity recognition (McCallum and Li, 2003).
---------------------------------------------------
W07-2218:77	264:279	Undirected graphical models, in particular Conditional Random Fields, are the standard tools for shallow parsing (Sha and Pereira, 2003).
---------------------------------------------------
P05-1056:78	42:170	452 2 CRF Model Description A CRF is a random field that is globally conditioned on an observation sequence O. CRFs have been successfully used for a variety of text processing tasks (Lafferty et al. , 2001; Sha and Pereira, 2003; McCallum and Li, 2003), but they have not been widely applied to a speech-related task with both acoustic and textual knowledge sources.
---------------------------------------------------
W05-1514:79	39:207	of machine learning techniques that have been developed for sequence labeling problems such as Hidden Markov Models, sequential classification with SVMs (Kudo and Matsumoto, 2001), and Conditional Random Fields (Sha and Pereira, 2003).
---------------------------------------------------
P09-1102:80	115:232	Sha and Pereira (2003) originally proposed this concept of implementing transition restrictions.
---------------------------------------------------
P03-1064:81	17:231	Many machine learning techniques have been successfully applied to chunking tasks, such as Regularized Winnow (Zhang et al. , 2001), SVMs (Kudo and Matsumoto, 2001), CRFs (Sha and Pereira, 2003), Maximum Entropy Model (Collins, 2002), Memory Based Learning (Sang, 2002) and SNoW (Munoz et al. , 1999).
---------------------------------------------------
J05-1003:82	532:603	The function in equation (28) can be optimized using variants of gradient descent, which in practice require tens or at most hundreds of passes over the training data (see, e.g., Sha and Pereira 2003).
---------------------------------------------------
J05-1003:83	529:603	See also Sha and Pereira (2003) for more recent work on CRFs.
---------------------------------------------------
H05-1099:84	89:185	Replacing the POS tags in the input text with Brill POS tags before the 5We follow Sha and Pereira (2003) in deriving the NP constituents from the CoNLL-2000 data sets, by replacing all nonNP shallow tags with the outside (O) tag.
---------------------------------------------------
H05-1099:85	94:185	Sha and Pereira (2003) noted that the difference between their perceptron and CRF results was not significant, and our performance falls between the two, thus replicating their result within noise.
---------------------------------------------------
H05-1099:86	82:185	We follow Collins (2002) and Sha and Pereira (2003) in using section 21 as a heldout set.
---------------------------------------------------
H05-1099:87	71:185	To this end, we replicated the NP-chunker described in Sha and Pereira (2003) and trained it as either an NP-chunker or with the tagset extended to classify all 11 phrase types included in the CoNLL-2000 task (Sang and Buchholz, 2000).
---------------------------------------------------
H05-1099:88	103:185	3.2 Combining Finite-State and Context-Free Parsers It is likely true that a context-free parser which has been optimized for global parse accuracy will, on occasion, lose some shallow parse accuracy to satisfy global structure constraints that do not constrain 7Sha and Pereira (2003) reported the Kudo and Matsumoto (2001) performance on the NP-Chunking task to be 94.39 and to be the best reported result on this task.
---------------------------------------------------
H05-1099:89	79:185	task, originally introduced in Ramshaw and Marcus (1995) and also described in (Collins, 2002; Sha and Pereira, 2003), brackets just base NP constituents5.
---------------------------------------------------
H05-1099:90	65:185	Both the Bikel (2004) imple789 System NP-Chunking CoNLL-2000 Li & Roth task SPRep averaged perceptron 94.21 93.54 95.12 Kudo and Matsumoto (2001) 94.22 93.91 Sha and Pereira (2003) CRF 94.38 Voted perceptron 94.09 Zhang et al.
---------------------------------------------------
H05-1099:91	75:185	See (Sha and Pereira, 2003) for more details on this approach.
---------------------------------------------------
P09-1056:92	96:216	Our chunker follows the system described by Sha and Pereira (2003).
---------------------------------------------------
P05-1044:93	2:262	c2005 Association for Computational Linguistics Contrastive Estimation: Training Log-Linear Models on Unlabeled Data Noah A. Smith and Jason Eisner Department of Computer Science / Center for Language and Speech Processing Johns Hopkins University, Baltimore, MD 21218 USA {nasmith,jason}@cs.jhu.edu Abstract Conditional random fields (Lafferty et al. , 2001) are quite effective at sequence labeling tasks like shallow parsing (Sha and Pereira, 2003) and namedentity extraction (McCallum and Li, 2003).
---------------------------------------------------
D07-1083:94	116:316	For the baseline method, we performed a conditional random field (CRF), which is exactly the same training procedure described in (Sha and Pereira, 2003) with L-BFGS.
---------------------------------------------------
P08-1024:95	88:227	We use a zero-mean Gaussian prior with the probability density function p0(k)  expparenleftbig2k/22parenrightbig.2 This results in the following log-likelihood objective and corresponding gradient: L= summationdisplay (e,f)D logp(e|f) + summationdisplay k logp0(k) (4) L k = Ep(d|e,f)[hk]Ep(e|f)[hk] k 2 (5) In order to train the model, we maximise equation (4) using L-BFGS (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
H05-1124:96	133:189	For the NP chunking experiments we used the standard CoNLL 2000 data set (Kudo and Matsumoto, 2001; Sha and Pereira, 2003) using the predicate set defined by Sha and Pereira (2003).
---------------------------------------------------
H05-1124:97	12:189	Models of that form include hidden Markov models (Rabiner, 1989; Bikel et al. , 1999) as well as discriminative tagging models based on maximum entropy classification (Ratnaparkhi, 1996; McCallum et al. , 2000), conditional random fields (Lafferty et al. , 2001; Sha and Pereira, 2003), and large-margin techniques (Kudo and Matsumoto, 2001; Taskar et al. , 2003).
---------------------------------------------------
H05-1124:98	37:189	Sequential tagging with the BIO tag set has proven quite accurate for shallow parsing and named entity extraction tasks (Kudo and Matsumoto, 2001; Sha and Pereira, 2003; Tjong Kim Sang and De Meulder, 2003).
---------------------------------------------------
P07-1120:99	45:209	The shallow parser is trained using the perceptron algorithm, with a feature set nearly identical to that from Sha and Pereira (2003), and achieves comparable performance to that paper.
---------------------------------------------------
P06-1009:100	55:213	We use L-BFGS, an iterative quasi-Newton optimisation method, which performs well for training log-linear models (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
P04-1014:101	124:189	However, we found that GIS converged extremely slowly; this is in line with other recent results in the literature applying GIS to globally optimised models such as conditional random fields, e.g. Sha and Pereira (2003).
---------------------------------------------------
W06-3004:102	94:171	In this section, we describe howwehaveusedanewsetofuserinteractionmodels (UIM) in conjunction with a relevance classifier based on conditional random fields (CRF) (McCallum, 2003; Sha and Pereira, 2003) in order to improve the relevance of the QUAB suggestions that FERRET returnsinresponsetoausersquery.
---------------------------------------------------
I08-1050:103	67:142	Conditional Random Fields (CRFs) have been successfully applied to various NLP tasks including part-of-speech tagging (Lafferty et al., 2001) and shallow parsing (Sha and Pereira, 2003).
---------------------------------------------------
C08-1094:104	63:177	tures similar to those used for NP chunking in Sha and Pereira (2003), including surrounding POStags (provided by a separately trained log linear POS-tagger) and surrounding words, up to 2 before and 2 after the current word position.
---------------------------------------------------
P05-1063:105	98:335	A number of resultse.g. , in Sha and Pereira (2003) and Roark et al.
---------------------------------------------------
P06-2098:106	86:155	Despite this restriction, the voted perceptron is known for its performance (Sha and Pereira, 2003).
---------------------------------------------------
P06-2098:107	113:155	However, on sequence labeling tasks CRFs have shown very good performance (Lafferty et al, 2001; Sha and Pereira, 2003), and we will use them for the baseline comparison.
---------------------------------------------------
W08-0303:108	35:239	In most applications like (Tseng et al., 2005; Sha and Pereira, 2003), a sequential model is used.
---------------------------------------------------
N09-2062:109	127:143	NP F1 Prc Rcl F1 AZ05 94.70 94.57 94.20 94.39 KM01 94.39 93.89 93.92 93.91 I-T-W-W-O 94.44 93.72 93.91 93.81 CM03 94.41 94.19 93.29 93.74 SP03 94.38 Mc03 93.96 AZ0593.83 93.37 93.60 ZDJ02 93.89 93.54 93.60 93.57 Table3:TestsetresultsforAndoandZhang(2005),Kudo and Matsumoto (2001), our I-T-W-W-O model, Carreras and M`arquez (2003), Sha and Pereira (2003), McCallum (2003), Zhang et al.
---------------------------------------------------
E06-1038:110	146:236	An important distinction we should make is the notion of supported versus unsupported features (Sha and Pereira, 2003).
---------------------------------------------------
N06-2024:111	6:86	Recently, sequential learning methods, such as hidden Markov models (HMMs) and conditional random fields (CRFs), have been used successfully for a number of applications, including NER (Sha and Pereira, 2003; Pinto et al. , 2003; Mccallum and Lee, 2003).
---------------------------------------------------
W06-1620:112	90:273	We use L-BFGS, an iterative quasi-Newton optimisation method, which performs well for training log-linear models (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
P05-1012:113	7:209	However, generative models make complicated and poorly justified independence assumptions and estimations, so we might expect better performance from discriminatively trained models, as has been shown for other tasks like document classification (Joachims, 2002) and shallow parsing (Sha and Pereira, 2003).
---------------------------------------------------
P08-1081:114	113:251	(See Section 3.4 for more about CRFs) Linear CRF model has been successfully applied in NLP and text mining tasks (McCallum and Li, 2003; Sha and Pereira, 2003).
---------------------------------------------------
P06-2013:115	169:216	The SVMs and CRFs approaches were used in the experiments because they provided good performance in chunking(Kudo and Matsumoto, 2001)(Sha and Pereira, 2003).
---------------------------------------------------
P06-2013:116	58:216	We only describe these models briefly since full details are presented elsewhere(Kudo and Matsumoto, 2001; Sha and Pereira, 2003; Ramshaw and Marcus, 1995; Sang, 2002).
---------------------------------------------------
P06-2013:117	68:216	Sha and Pereira(Sha and Pereira, 2003) showed that stateof-the-art results can be achieved using CRFs in English chunking.
---------------------------------------------------
D07-1064:118	53:279	Collinss method, like the linear-chain conditional random fields (CRFs) (Lafferty et al. , 2001; Sha and Pereira, 2003), seeks for a complete path from the initial vertex to the terminal using the Viterbi algorithm.
---------------------------------------------------
D07-1064:119	109:279	The external function f(x,y) returns a vector (called the global feature vector in (Sha and Pereira, 2003)) of the number of feature occurrences along the alignment path y. In the beginning (line 5 in the figure) of the inner loop, the target path (alignment) input: Set of examples S = {(x i,Y i )} Iteration cutoff T output: Averaged weight vector w 1: w  0; w  0 2: for each (x i,Y i )  S do 3: g i  (1/|Y i |) yY i f(x i,y) 4: end for 5: for t  1T do 6: w  0 7: for each (x i,Y i )  S do 8: y prime  argmax yA(x i ) w f(x i,y) 9: Convert y prime into its box representation Y prime 10: g prime  (1/|Y prime i |) yY prime i f(x i,y) 11: f  g i g prime 12: w w+f 13: end for 14: if w = 0 then 15: return w 16: end if 17: w  w+w 18: w  [(t 1) w+w]/t 19: end for 20: return w Figure 7: Box-based algorithm is recomputed with the current weight vector w.The argmax in lines 5 and 6 can be computed efficiently (O(n 2 ), where n is the number of words in x)byrunning a pass of the Viterbi algorithm in the edit graph for x. The weight vector w varies between iterations, and so does the most likely alignment with respect to w. Hence the recomputation in line 5 is needed.
---------------------------------------------------
P07-1095:120	112:207	Extended Feature Set Sha and Pereira (2003) applied a conditional random field to the NP chunking task, achieving excellent results.
---------------------------------------------------
P07-1095:121	121:207	Using the original HMM feature set and the extended feature set, we trained four models that can use arbitrary features: conditional random fields (a near-replication of Sha and Pereira, 2003), maximum entropy Markov models (MEMMs; McCallum et al. , 2000), pseudolikelihood (Besag, 1975; see Toutanova et al. , 2003, for a tagging application), and our M-estimator with the HMM as q0.
---------------------------------------------------
P05-1003:122	8:159	1 Introduction In recent years, conditional random fields (CRFs) (Lafferty et al. , 2001) have shown success on a number of natural language processing (NLP) tasks, including shallow parsing (Sha and Pereira, 2003), named entity recognition (McCallum and Li, 2003) and information extraction from research papers (Peng and McCallum, 2004).
---------------------------------------------------
P05-1003:123	27:159	Malouf (2002) and Sha and Pereira (2003) show that gradient-based algorithms, particularly limited memory variable metric (LMVM), require much less time to reach convergence, for some NLP tasks, than the iterative scaling methods (Della Pietra et al. , 1997) previously used for log-linear optimisation problems.
---------------------------------------------------
W06-2918:124	27:203	For our experiments we use the limited memory variable metric (LMVM) (Sha and Pereira, 2003) routine, which has become the standard algorithm for CRF training with a likelihood-based objective function.
---------------------------------------------------
W06-2918:125	8:203	1 Introduction In recent years discriminative probabilistic models have been successfully applied to a number of information extraction tasks in natural language processing (NLP), such as named entity recognition (NER) (McCallum and Li, 2003), noun phrase chunking (Sha and Pereira, 2003) and information extraction from research papers (Peng and McCallum, 2004).
---------------------------------------------------
W06-0107:126	118:185	CRFs have been shown to perform well on a number of NLP problems such as shallow parsing (Sha and Pereira, 2003), table extraction (Pinto et al. , 2003), and named entity recognition (McCallum and Li, 2003).
---------------------------------------------------
D09-1071:127	200:228	For the experiments with shallow parsing we used the CRF++ toolkit3 which has an efficient implementation of the model introduced by Sha & Pereira (2003) for this task.
---------------------------------------------------
D07-1015:128	123:443	We train the model using the approach described by Sha and Pereira (2003).
---------------------------------------------------
