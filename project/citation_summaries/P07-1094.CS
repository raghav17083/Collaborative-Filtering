D08-1109:1	169:232	We used our own implementation after verifying that its performance on WSJ was identical to that reported in (Goldwater and Griffiths, 2007).
---------------------------------------------------
D08-1109:2	50:232	Recent advances in these approaches include the use of a fully Bayesian HMM (Johnson, 2007; Goldwater and Griffiths, 2007).
---------------------------------------------------
D08-1109:3	132:232	We also evaluate tagger performance when only incomplete dictionaries are available (Smith and Eisner, 2005; Goldwater and Griffiths, 2007).
---------------------------------------------------
D08-1109:4	150:232	The annotation distinguishes between 13 parts-ofspeech, of which 11 are common for all languages 1046 Random Monolingual Unsupervised Monolingual Supervised Trigram Entropy EN 56.24 90.71 96.97 1.558 BG 82.68 88.88 96.96 1.708 SL 84.70 87.41 97.31 1.703 SR 83.41 85.05 96.72 1.789 Table 1: Monolingual tagging accuracy for English, Bulgarian, Slovene, and Serbian for two unsupervised baselines (random tag selection and a Bayesian HMM (Goldwater and Griffiths, 2007)) as well as a supervised HMM.
---------------------------------------------------
D08-1109:5	196:232	Slovene, for instance, achieves a large improvement when paired with Serbian (+7.7), a closely related Slavic language, but only a minor improvement when coupled 5The accuracy of the monolingual English tagger is relatively high compared to the 87% reported by (Goldwater and Griffiths, 2007) on the WSJ corpus.
---------------------------------------------------
D08-1109:6	162:232	4 Monolingual Baseline As our monolingual baseline we use the unsupervised Bayesian HMM model of Goldwater and Griffiths (2007) (BHMM1).
---------------------------------------------------
P08-1100:7	11:214	Bayesian approaches can also improve performance (Goldwater and Griffiths, 2007; Johnson, 2007; Kurihara and Sato, 2006).
---------------------------------------------------
N09-1069:8	187:198	(2006) and Goldwater and Griffiths (2007).
---------------------------------------------------
N09-1010:9	178:223	Monolingual and bilingual baselines We reimplemented the Bayesian HMM model of Goldwater and Griffiths (2007) (BHMM1) as our monolingual baseline.
---------------------------------------------------
N09-1010:10	167:223	Results are given for a monolingual Bayesian HMM (Goldwater and Griffiths, 2007), a bilingual model (Snyder et al., 2008), and the multilingual model presented here.
---------------------------------------------------
N09-1010:11	135:223	Each integral can be computed in closed form using multinomial-Dirichlet conjugacy (and by making the above-mentioned simplifying assumption that all other tags were generated separately by their transition and superlingual 87 parameters), just as in the monolingual Bayesian HMM of (Goldwater and Griffiths, 2007).
---------------------------------------------------
C08-1008:12	93:175	Dirichlet priors can be used to bias HMMs toward more skewed distributions (Goldwater and Griffiths, 2007; Johnson, 2007), which is especially useful in the weakly supervised setting consideredhere.
---------------------------------------------------
C08-1008:13	164:175	Other work aims to do truly unsupervised learning of taggers, such as Goldwater and Griffiths (2007) and Johnson (2007).
---------------------------------------------------
D08-1036:14	129:151	The largest corpus that Goldwater and Griffiths (2007) studied contained 96,000 words, while Johnson (2007) used all of the 1,173,766 words in the full Penn WSJ treebank.
---------------------------------------------------
D08-1036:15	48:151	t |   Dir() t | prime  Dir(prime) A multinomial  is distributed according to the Dirichlet distribution Dir() iff: P( | )  mproductdisplay j=1 j1j In our experiments we set  and prime to the uniform values (i.e., all components have the same value  or prime), but it is possible to estimate these as well (Goldwater and Griffiths, 2007).
---------------------------------------------------
D08-1036:16	79:151	The samplers that Goldwater and Griffiths (2007) and Johnson (2007) describe are pointwise collapsed Gibbs samplers.
---------------------------------------------------
D08-1036:17	98:151	Goldwater and Griffiths (2007) proposed an information-theoretic measure known as the Variation of Information (VI) described by Meila (2003) as an evaluation of an unsupervised tagging.
---------------------------------------------------
D08-1036:18	31:151	On the other hand, Goldwater and Griffiths (2007) reported that the same kind of Gibbs sampler produced much better results than EM on their unsupervised POS tagging task.
---------------------------------------------------
D08-1036:19	139:151	On small data sets all of the Bayesian estimators strongly outperform EM (and, to a lesser extent, VB) with respect to all of our evaluation measures, confirming the results reported in Goldwater and Griffiths (2007).
---------------------------------------------------
D08-1036:20	125:151	The studies presented by Goldwater and Griffiths (2007) and Johnson (2007) differed in the number of states that they used.
---------------------------------------------------
D08-1036:21	28:151	Monte Carlo sampling methods and Variational Bayes are two kinds of approximate inference methods that have been applied to Bayesian inference of unsupervised HMM POS taggers (Goldwater and Griffiths, 2007; Johnson, 2007).
---------------------------------------------------
D08-1036:22	126:151	Goldwater and Griffiths (2007) evaluated against the reduced tag set of 17 tags developed by Smith and Eisner (2005), while Johnson (2007) evaluated against the full Penn Treebank tag set.
---------------------------------------------------
D07-1031:23	118:275	However, in the ad hoc approach the expected count plus1 may be less than zero, resulting in a value of zero for the corresponding parameter (Johnson et al. , 2007; Goldwater and Griffiths, 2007).
---------------------------------------------------
D07-1031:24	24:275	Most previous work exploiting unsupervised training data for inferring POS tagging models has focused on semi-supervised methods in the in which the learner is provided with a lexicon specifying the possible tags for each word (Merialdo, 1994; Smith and Eisner, 2005; Goldwater and Griffiths, 2007) or a small number of prototypes for each POS (Haghighi and Klein, 2006).
---------------------------------------------------
D07-1031:25	40:275	Goldwater and Griffiths (2007) propose using the Variation of Information (VI) metric described by Meila (2003).
---------------------------------------------------
D07-1031:26	93:275	The application of MCMC techniques, including Gibbs sampling, to HMM inference problems is relatively well-known: see Besag (2004) for a tutorial introduction and Goldwater and Griffiths (2007) for an application of Gibbs sampling to HMM inference for semi300 supervised and unsupervised POS tagging.
---------------------------------------------------
P09-1088:27	126:209	Instead we perform inference over the hyperparameters following Goldwater and Griffiths (2007) by defining a vague gamma prior on each concentration parameter, x  Gamma(104,104).
---------------------------------------------------
E09-1041:28	19:242	354 supervised induction techniques that have been successfully developed for English (e.g., Schutze (1995), Clark (2003)), including the recentlyproposed prototype-driven approach (Haghighi and Klein, 2006) and Bayesian approach (Goldwater and Griffiths, 2007).
---------------------------------------------------
W09-1121:29	12:267	Recent NLP problems addressed by clustering include POS induction (Clark, 2003; Goldwater and Griffiths, 2007), word sense disambiguation (Shin and Choi, 2004), semantic role labeling (Baldewein et al., 2004), pitch accent type disambiguation (Levow, 2006) and grammar induction (Klein, 2005).
---------------------------------------------------
C08-1042:30	19:202	In the first, tag induction systems are allowed the use of a tagging dictionary, which specifies for each word a set of possible parts-of-speech (Merialdo, 1994; Smith and Eisner, 2005; Goldwater and Griffiths, 2007).
---------------------------------------------------
C08-1042:31	49:202	Johnson (2007) evaluates both estimation techniques on the Bayesian bitag model; Goldwater and Griffiths (2007) emphasize the advantage in the MCMC approach of integrating out the HMM parameters in a tritag model, yielding a tagging supported by many different parameter settings.
---------------------------------------------------
C08-1042:32	88:202	Goldwater and Griffiths (2007) propose using the Variation of Information of Meila (2003): V I(TG;TI) = H(TG|TI) + H(TI|TG) VI represents the change in information when going from one clustering to another.
---------------------------------------------------
C08-1042:33	45:202	For an HMM with a set of states T and a set of output symbols V : t  T t  Dir(1,|T|) (1) t  T t  Dir(1,|V |) (2) ti|ti1, ti1  Multi(ti1) (3) wi|ti, ti  Multi(ti) (4) One advantage of the Bayesian approach is that the prior allows us to bias learning toward sparser structures, by setting the Dirichlet hyperparameters , to a value less than one (Johnson, 2007; Goldwater and Griffiths, 2007).
---------------------------------------------------
C08-1042:34	3:202	1 Introduction There has been a great deal of recent interest in the unsupervised discovery of syntactic structure from text, both parts-of-speech (Johnson, 2007; Goldwater and Griffiths, 2007; Biemann, 2006; Dasgupta and Ng, 2007) and deeper grammatical structure like constituency and dependency trees (Klein and Manning, 2004; Smith, 2006; Bod, 2006; Seginer, 2007; Van Zaanen, 2001).
---------------------------------------------------
C08-1042:35	47:202	There is evidence that this leads to better performance on some part-of-speech induction metrics (Johnson, 2007; Goldwater and Griffiths, 2007).
---------------------------------------------------
P09-1056:36	197:216	HMMs have been used many times for POS tagging and chunking, in supervised, semisupervised, and in unsupervised settings (Banko and Moore, 2004; Goldwater and Griffiths, 2007; Johnson, 2007; Zhou, 2004).
---------------------------------------------------
W09-0905:37	70:202	Due to its popularity for unsupervised POS induction research (e.g., Goldberg et al., 2008; Goldwater and Griffiths, 2007; Toutanova and Johnson, 2008) and its often-used tagset, for our initial research, we use the Wall Street Journal (WSJ) portion of the Penn Treebank (Marcus et al., 1993), with 36 tags (plus 9 punctuation tags), and we use sections 00-18, leaving held-out data for future experiments.4 Defining frequent frames as those occurring at 4Even if we wanted child-directed speech, the CHILDES database (MacWhinney, 2000) uses coarse POS tags.
---------------------------------------------------
W09-0905:38	120:202	For example, the PTB-17 mapping (Smith and Eisner, 2005) is commonly used for evaluating category induction (Goldwater and Griffiths, 2007; Toutanova and Johnson, 2008), yet it loses distinctions for 2038 words.
---------------------------------------------------
W09-0905:39	19:202	For many evaluations, POS tags have been mapped to a smaller tagset (e.g., Goldwater and Griffiths, 2007; Toutanova and Johnson, 2008), but there have been few criteria for evaluating the quality of these mappings.
---------------------------------------------------
P09-1057:40	30:223	Bayesian method with sparse priors (Goldwater and Griffiths, 2007) 86.8 5.
---------------------------------------------------
P09-1057:41	20:223	The classic approach (Merialdo, 1994) is expectation-maximization (EM), where we estimate grammar and dictionary probabilities in order to maximize the probability of the observed word sequence: P(w1wn) = summationdisplay t1tn P(t1tn)P(w1wn|t1tn)  summationdisplay t1tn nproductdisplay i=1 P(ti|ti2 ti1)P(wi|ti) Goldwater and Griffiths (2007) report 74.5% accuracy for EM with a 3-gram tag model, which we confirm by replication.
---------------------------------------------------
P09-1057:42	170:223	6 Smaller Tagset and Incomplete Dictionaries Previously, researchers working on this task have also reported results for unsupervised tagging with a smaller tagset (Smith and Eisner, 2005; Goldwater and Griffiths, 2007; Toutanova and Johnson, 2008; Goldberg et al., 2008).
---------------------------------------------------
P09-1057:43	29:223	Bayesian method (Goldwater and Griffiths, 2007) 83.9 4b.
---------------------------------------------------
P09-1057:44	166:223	But previous results from Smith and Eisner (2005) and Goldwater and Griffiths (2007) show that their models do not benefit from using more unlabeled training data.
---------------------------------------------------
P09-1057:45	215:223	We also note that it might be possible to replicate our models in a Bayesian framework similar to that proposed in (Goldwater and Griffiths, 2007).
---------------------------------------------------
N09-1005:46	19:180	Goldwater and Griffiths, 2007), and we are also inspired by cryptanalysiswe view a corpus of foreign terms as a code for English, and we attempt to break the code.
---------------------------------------------------
P08-1085:47	15:220	In recent work, researchers try to address these deficiencies by using dictionaries with unfiltered POS-tags, and testing the methods on diluted dictionaries  in which many of the lexical entries are missing (Smith and Eisner, 2005) (SE), (Goldwater and Griffiths, 2007) (GG), (Toutanova and Johnson, 2008) (TJ).
---------------------------------------------------
P08-1012:48	42:198	3 Variational Bayes for ITG Goldwater and Griffiths (2007) and Johnson (2007) show that modifying an HMM to include a sparse prior over its parameters and using Bayesian estimation leads to improved accuracy for unsupervised part-of-speech tagging.
---------------------------------------------------
W08-2112:49	232:249	Recent models that apply Bayesian approaches to PoS tagging are not incremental and assume a fixed number of tags (Goldwater and Griffiths, 2007; Toutanova and Johnson, 2008).
---------------------------------------------------
E09-1042:50	2:182	c2009 Association for Computational Linguistics Weakly Supervised Part-of-Speech Tagging for Morphologically-Rich, Resource-Scarce Languages Kazi Saidul Hasan and Vincent Ng Human Language Technology Research Institute University of Texas at Dallas Richardson, TX 75083-0688 {saidul,vince}@hlt.utdallas.edu Abstract This paper examines unsupervised approaches to part-of-speech (POS) tagging for morphologically-rich, resource-scarce languages, with an emphasis on Goldwater and Griffithss (2007) fully-Bayesian approach originally developed for English POS tagging.
---------------------------------------------------
E09-1042:51	39:182	Similar to Goldwater and Griffiths (2007) and Johnson (2007), Toutanova and Johnson (2007) also use Bayesian inference for POS tagging.
---------------------------------------------------
E09-1042:52	37:182	As mentioned before, Goldwater and Griffiths (2007) have recently proposed an unsupervised fully-Bayesian POS tagging framework that operates by integrating over the possible parameter values instead of fixing a set of parameter values for unsupervised sequence learning.
---------------------------------------------------
E09-1042:53	94:182	366 P(ti|ti,w,,)  n(ti,wi) +n ti +Wti .n(ti2,ti1,ti) +n (ti2,ti1) +T .n(ti1,ti,ti+1) +I(ti2 = ti1 = ti = ti+1) +n (ti1,ti) +I(ti2 = ti1 = ti)+T .n(ti,ti+1,ti+2) +I(ti2 = ti = ti+2,ti1 = ti+1) +I(ti1 = ti = ti+1 = ti+2) +n (ti,ti+1) +I(ti2 = ti,ti1 = ti+1) +I(ti1 = ti = ti+1)+T Figure 2: The sampling distribution for ti (taken directly from Goldwater and Griffiths (2007)).
---------------------------------------------------
E09-1042:54	13:182	More recently, a fully-Bayesian approach to unsupervised POS tagging has been developed by Goldwater and Griffiths (2007) [henceforth G&G] as a viable alternative to the traditional maximumlikelihood-based HMM approach.
---------------------------------------------------
D09-1072:55	11:226	For example, (Smith&Eisner, 2005) proposes contrastive estimation (CE) for log-linear models (CRF), achieving the current state-of-the-art performance of 90.4%; (Goldwater&Griffiths, 2007) applies a Bayesian approach to improve maximumlikelihood estimation (MLE) for training generative models (HMM).
---------------------------------------------------
D09-1072:56	169:226	CRF/CE (Smith&Eisner, 2005) and BHMM2 (Goldwater&Griffiths, 2007) have been discussed briefly in the introduction.
---------------------------------------------------
W09-0715:57	65:185	Goldwater and Griffiths (2007) noted that most recent approaches to this problem aim to identify the set of attributes that maximizes some target function (Maximum Likelihood Estimation), and then to select the values of these attributes based on the representation of the model.
---------------------------------------------------
W09-0437:58	204:217	Accounting for sparsity explicitly has achieved significant improvements in other areas such as in part of speech tagging (Goldwater and Griffiths, 2007).
---------------------------------------------------
D09-1071:59	18:228	One potential solution is to add a small amount of supervision as in Goldwater & Griffiths (2007) who assume a dictionary of frequent words associated with possible PoS tags extracted from a labeled corpus.
---------------------------------------------------
D09-1071:60	133:228	Recent work (Goldwater and Griffiths, 2007; Johnson, 2007; Gao and Johnson, 2008) on this task explored a variety of methodologies to address this issue.
---------------------------------------------------
D09-1071:61	12:228	Recent work (Johnson, 2007; Goldwater and Griffiths, 2007; Gao and Johnson, 2008) explored the task of part-of-speech tagging (PoS) using unsupervised Hidden Markov Models (HMMs) with encouraging results.
---------------------------------------------------
D09-1071:62	155:228	The fact that different authors use different versions of the same gold standard to evaluate similar experiments (e.g. Goldwater & Griffiths (2007) versus Johnson (2007)) supports this claim.
---------------------------------------------------
D09-1071:63	31:228	This answers an open problem from Goldwater & Griffiths (2007).
---------------------------------------------------
D09-1071:64	22:228	Johnson (2007) reports results for different numbers of hidden states but it is unclear how to make this choice a priori, while Goldwater & Griffiths (2007) leave this question as future work.
---------------------------------------------------
