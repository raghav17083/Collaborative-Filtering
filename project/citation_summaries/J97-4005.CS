J07-4004:1	80:948	A different approach is proposed by Abney (1997), who develops log-linear models for attribute-value grammars, such as Head-driven Phrase Structure Grammar (HPSG).
---------------------------------------------------
P02-1042:2	45:157	3 The Probability Model The DAG-like nature of the dependency structures makes it difficult to apply generative modelling techniques (Abney, 1997; Johnson et al. , 1999), so we have defined a conditional model, similar to the model of Collins (1996) (see also the conditional model in Eisner (1996b)).
---------------------------------------------------
J99-1004:3	71:455	Among the most widely studied is the Gibbs distribution (Mark, Miller, and Grenander 1996; Mark et al. 1996; Mark 1997; Abney 1997).
---------------------------------------------------
P02-1043:4	117:171	As is well known (Abney, 1997), DAG-like dependencies cannot in general be modeled with a generative approach of the kind taken here 3 . 5.6 Comparison with Clark et al.
---------------------------------------------------
P02-1036:5	35:163	The following section reviews stochastic uni cation grammars (Abney, 1997) and the statistical quantities required for ef ciently estimating such grammars from parsed training data (Johnson et al. , 1999).
---------------------------------------------------
P02-1036:6	11:163	Abney (1997) pointed out that the non-contextfree dependencies of a uni cation grammar require stochastic models more general than Probabilistic Context-Free Grammars (PCFGs) and Markov Branching Processes, and proposed the use of loglinear models for de ning probability distributions over the parses of a uni cation grammar.
---------------------------------------------------
W03-0401:7	127:185	Using this model, we can assign consistent probabilities to parsing results with complex structures, such as ones represented with feature structures (Abney, 1997; Johnson et al. , 1999).
---------------------------------------------------
W03-0401:8	10:185	For example, when we apply a unification-based grammar, LPCFG-like modeling results in an inconsistent probability model because the model assigns probabilities to parsing results not allowed by the grammar (Abney, 1997).
---------------------------------------------------
W03-0401:9	84:185	In particular, when we apply Feature-Based LTAG (FBLTAG), the above probability is no longer consistent because of the non-local constraints caused by feature unification (Abney, 1997).
---------------------------------------------------
P04-1041:10	152:167	It is well known (Abney, 1997) that PCFG-type approximations to unification grammars can yield inconsistent probability models due to loss of probability mass: the parser successfully returns the highest ranked parse tree but the constraint solver cannot resolve the f-equations (generated in the pipeline or hidden in the integrated model) and the probability mass associated with that tree is lost.
---------------------------------------------------
C00-1085:11	6:252	1 Introduction Abney showed that attribute-value grammars cannot be modelled adequately using statistical techniques which assume that statistical dependencies are accidental (Ablmy, 1997).
---------------------------------------------------
W08-1302:12	30:177	Thus, we seek the model with minimum KL divergence from the uniform distribution, which means we search model p with maximum entropy (uncertainty) subject to given constraints (Abney, 1997).
---------------------------------------------------
W08-1302:13	23:177	2 Background: MaxEnt Models Maximum Entropy (MaxEnt) models are widely used in Natural Language Processing (Berger et al., 1996; Ratnaparkhi, 1997; Abney, 1997).
---------------------------------------------------
W02-2018:14	7:125	In natural language processing, recent years have seen ME techniques used for sentence boundary detection, part of speech tagging, parse selection and ambiguity resolution, and stochastic attribute-value grammars, to name just a few applications (Abney, 1997; Berger et al. , 1996; Ratnaparkhi, 1998; Johnson et al. , 1999).
---------------------------------------------------
A00-2021:15	35:132	As Abney (1997) showed, under these circumstances the relative frequency estimator is in general inconsistent, even if one restricts attention to rule features.
---------------------------------------------------
A00-2021:16	6:132	Recent work has shown how to define probability distributions over the parses of UBGs (Abney, 1997) and efficiently estimate and use conditional probabilities for parsing (Johnson et al. , 1999).
---------------------------------------------------
P06-1130:17	172:183	Simple PCFGbased models, while effective and computationally efficient, can only provide approximations to LFG and similar constraint-based formalisms (Abney, 1997).
---------------------------------------------------
P06-1130:18	42:183	In the in1The resources are approximations in that (i) they do not enforce LFG completeness and coherence constraints and (ii) PCFG-based models can only approximate LFG and similar constraint-based formalisms (Abney, 1997).
---------------------------------------------------
W00-0709:19	9:132	1 Introduction The maximum entropy technique of statistical modeling using random fields has proved to be an effective way of dealing with a variety of linguistic phenomena, in particular where modeling of attribute-valued grammars (AVG's) is concerned (Abney, 1997).
---------------------------------------------------
J05-1003:20	516:603	This is a problem for parameter estimation, in which an estimate of the denominator is required, and Monte Carlo methods have been proposed (Della Pietra, Della Pietra, and Lafferty 1997; Abney 1997; Rosenfeld 1997) as a technique for estimation of this value.
---------------------------------------------------
J05-1003:21	42:603	In particular, previous work (Ratnaparkhi, Roukos, and Ward 1994; Abney 1997; Della Pietra, Della Pietra, and Lafferty 1997; Johnson et al. 1999; Riezler et al. 2002) has investigated the use of Markov random fields (MRFs) or log-linear models as probabilistic models with global features for parsing and other NLP tasks.
---------------------------------------------------
J05-1003:22	513:603	6.2 Joint Log-Linear Models Abney (1997) describes the application of log-linear models to stochastic headdriven phrase structure grammars (HPSGs).
---------------------------------------------------
P05-1044:23	91:262	Because CRF implies CL estimation, we use the term WFSA. 356 putation, like random sampling (see, e.g., Abney, 1997), will not help to avoid this difficulty; in addition, convergence rates are in general unknown and bounds difficult to prove.
---------------------------------------------------
W01-0714:24	143:226	The chance of a pair a8a10a2a9a42a17a3a36a11 being a constituent is a57 a8a15a83a53a45a2a9a42a17a3a36a11a28a27 a57 a8a15a83a81a45a2a16a11 a57 a8a15a83a53a45a3a32a11a17a33 a57 a8a15a83a97a11 and we score a tree a98 by the likelihood product of its judgements a83a81a8a10a2a28a42a89a98a99a11 . The best tree is then a100a74a101a103a102a28a104a6a100a34a105a107a106 a8a109a108 a67a110a69a111a72a112a26a71 a88 a24 a57 a8a15a83a81a8a10a2a28a42a89a98a99a11a97a45a2a28a42a17a3a32a11 As we are considering each pair independently from the rest of the parse, this model does not correspond to a generative model of the kind standardly associated with PCFGs, but can be seen as a random field over the possible parses, with the features being the sequences and contexts (see (Abney, 1997)).
---------------------------------------------------
W06-1619:25	46:171	Previous studies (Abney, 1997; Johnson et al. , 1999; Riezler et al. , 2000; Malouf and van Noord, 2004; Kaplan et al. , 2004; Miyao and Tsujii, 2005) defined a probabilistic model of unification-based grammars including HPSG as a log-linear model or maximum entropy model (Berger et al. , 1996).
---------------------------------------------------
J08-1002:26	34:587	This is because these grammar formalisms exploitfeaturestructurestorepresentlinguisticconstraints.Suchconstraintsareknown to introduce inconsistencies in probabilistic models estimated using simple relative frequency, as discussed in Abney (1997).
---------------------------------------------------
C02-1075:27	8:245	All approaches have in common that they try to model a probability distribution over the readings of the UBG, which can be used to rank the competing analyses of a given sentence; see, e.g., Briscoe and Carroll (1993), Eisele (1994), Brew (1995), Abney (1997), Goodman (1997), Bod and Kaplan (1998), Johnson et al.
---------------------------------------------------
C02-1075:28	12:245	Unfortunately, most of the proposed probability models are not mathematically clean in that the probabilities of all possible UBG readings do not sum to the value 1, a problem which is discussed intensively by Eisele (1994), Abney (1997), and Schmid (2002).
---------------------------------------------------
W09-0103:29	70:176	And as Abney (1997) showed, it is mathematically straight-forward to define probability distributions over the representations used by virtually any theory of grammar (even those of Chomskys Minimalism), which means that theoretically the arsenal of statistical methods for parsing and learning can be applied to any grammar just as well.
---------------------------------------------------
P04-1015:30	8:297	Examples of such techniques are Markov Random Fields (Ratnaparkhi et al. , 1994; Abney, 1997; Della Pietra et al. , 1997; Johnson et al. , 1999), and boosting or perceptron approaches to reranking (Freund et al. , 1998; Collins, 2000; Collins and Duffy, 2002).
---------------------------------------------------
W07-2208:31	43:192	Previous studies (Abney, 1997; Johnson et al. , 1999; Riezler et al. , 2000; Malouf and van Noord, 2004; Kaplan et al. , 2004; Miyao and Tsujii, 2005) defined a probabilistic model of unification-based grammars including HPSG as a log-linear model or maximum entropy model (Berger et al. , 1996).
---------------------------------------------------
W07-2208:32	10:192	This was overcome by a probabilistic model which provides probabilities of discriminating a correct parse tree among candidates of parse trees in a log-linear model or maximum entropy model (Berger et al. , 1996) with many features for parse trees (Abney, 1997; Johnson et al. , 1999; Riezler et al. , 2000; Malouf and van Noord, 2004; Kaplan et al. , 2004; Miyao and Tsujii, 2005).
---------------------------------------------------
W07-2219:33	111:344	(Abney, 1997)) and has the advantage of elegantly bypassing the issue of loosing probability mass to failed derivations due to unification failures.
---------------------------------------------------
P99-1069:34	9:148	On the other hand, as Abney (1997) points out, the context-sensitive dependencies that "unification-based" constraints introduce render the relative frequency estimator suboptimal: in general it does not maximize the likelihood and it is inconsistent.
---------------------------------------------------
P99-1069:35	11:148	Abney (1997) proposes a Markov Random Field or log linear model for SUBGs, and the models described here are instances of Abney's general framework.
---------------------------------------------------
P99-1069:36	52:148	Abney (1997) proposes a gradient ascent, based upon a Monte Carlo procedure for estimating E0(fj).
---------------------------------------------------
C04-1204:37	70:135	Following recent research about disambiguation models on linguistic grammars (Abney, 1997; Johnson et al. , 1999; Riezler et al. , 2002; Clark and Curran, 2003; Miyao et al. , 2003; Malouf and van Noord, 2004), we apply a log-linear model or maximum entropy model (Berger et al. , 1996) on HPSG derivations.
---------------------------------------------------
P02-1062:38	8:174	Examples of such techniques are Markov Random Fields (Abney 1997; Della Pietra et al. 1997; Johnson et al. 1999), and boosting algorithms (Freund et al. 1998; Collins 2000; Walker et al. 2001).
---------------------------------------------------
P05-1011:39	23:172	Such constraints are known 83 to introduce inconsistencies in probabilistic models estimated using simple relative frequency (Abney, 1997).
---------------------------------------------------
W03-1013:40	9:181	We would like the parsing model to include long-range dependencies, but this introduces problems for generative parsing models similar to those described by Abney (1997) for attribute-value grammars; hence Hockenmaier and Steedman do not include such dependencies in their model, and Clark et al. include the dependencies but use an inconsistent model.
---------------------------------------------------
W05-1511:41	38:178	Previous studies (Abney, 1997; Johnson et al. , 1999; Riezler et al. , 2000; Miyao et al. , 2003; Malouf and van Noord, 2004; Kaplan et al. , 2004; Miyao and Tsujii, 2005) defined a probabilistic model of unification-based grammars as a log-linear model or maximum entropy model (Berger et al. , 1996).
---------------------------------------------------
C02-2025:42	61:133	Abney (1997) notes important problems with the soundness of the approach when a unification-based grammar is actually determining the derivations, motivating the use of log-linear models (Agresti, 1990) for parse ranking that Johnson and colleagues further developed (Johnson, Geman, Canon, Chi, & Riezler, 1999).
---------------------------------------------------
E09-3005:43	46:229	The Maximum Entropy model (Berger et al., 1996; Ratnaparkhi, 1997; Abney, 1997) is a conditional model that assigns a probability to every possible parse  for a given sentence s. The model consists of a set of m feature functions fj() that describe properties of parses, together with their associated weights j. The denominator is a normalization term where Y (s) is the set of parses with yield s: p(|s;) = exp( summationtextm j=1 jfj())summationtext yY (s) exp( summationtextm j=1 jfj(y))) (1) The parameters (weights) j can be estimated efficiently by maximizing the regularized conditional likelihood of a training corpus (Johnson et al., 1999; van Noord and Malouf, 2005):  = argmax  logL()  summationtextm j=1  2j 22 (2) where L() is the likelihood of the training data.
---------------------------------------------------
W07-2207:44	9:381	Parse selection approaches for these frameworks often use discriminative Maximum Entropy (ME) models, where the probability of each parse tree, given an input string, is estimated on the basis of select properties (called features) of the tree (Abney, 1997; Johnson, Geman, Canon, Chi, & Riezler, 1999).
---------------------------------------------------
E09-1069:45	9:198	Previous studies (Abney, 1997; Johnson et al., 1999; Kaplan et al., 2004; Malouf and van Noord, 2004; Miyao and Tsujii, 2005; Riezler et al., 2000) defined a probabilistic model of unification-based grammars, including head-driven phrase structure grammar (HPSG), lexical functional grammar (LFG) and combinatory categorial grammar (CCG), as a maximum entropy model (Berger et al., 1996).
---------------------------------------------------
P05-1045:46	22:199	(1997) and Abney (1997).
---------------------------------------------------
