N04-1012:1	49:199	We use the limited memory variable metric algorithm (Malouf, 2002) to determine the weights.
---------------------------------------------------
N06-1019:2	41:159	The optimisation of the objective function is performed using the limited-memory BFGS numerical optimisation algorithm (Nocedal and Wright, 1999; Malouf, 2002), which requires calculation of the objective function and the gradient of the objective function at each iteration.
---------------------------------------------------
J07-4004:3	25:948	Log-linear models have previously been applied to statistical parsing (Johnson et al. 1999; Toutanova et al. 2002; Riezler et al. 2002; Malouf and van Noord 2004), but typically under the assumption that all possible parses for a sentence can be enumerated.
---------------------------------------------------
J07-4004:4	136:948	Statistical parsers have been developed for TAG (Chiang 2000; Sarkar and Joshi 2003), LFG (Riezler et al. 2002; Kaplan et al. 2004; Cahill et al. 2004), and HPSG (Toutanova et al. 2002; Toutanova, Markova, and Manning 2004; Miyao and Tsujii 2004; Malouf and van Noord 2004), among others.
---------------------------------------------------
J07-4004:5	347:948	Malouf (2002) gives a more thorough description of numerical optimization methods applied to log-linear models.
---------------------------------------------------
J07-4004:6	42:948	As Malouf (2002) demonstrates, general purpose numerical optimization algorithms such as BFGS can converge much faster than iterative scaling algorithms (including Improved Iterative Scaling; Della Pietra, Della Pietra, and Lafferty 1997).
---------------------------------------------------
P04-1007:7	80:275	This technique has been shown to be very effective in a variety of NLP tasks (Malouf, 2002; Wallach, 2002).
---------------------------------------------------
P06-3006:8	113:138	A detailed description of CRFs can be found in (Lafferty et al, 2001; Sha and Pereira, 2003; Malouf, 2002; Peng and McCallum, 2004).
---------------------------------------------------
W04-2413:9	42:115	We used theestimate software for estimation, which implements the LMVM algorithm (Malouf, 2002) and was kindly provided by Rob Malouf.
---------------------------------------------------
P03-1040:10	139:201	We use the toolkit7 developed by Malouf (2002).
---------------------------------------------------
I08-4011:11	25:106	Given a set of training examples, the log likelihood of the model with Gaussian prior (Chen and Rosenfeld, 1999) has the form 82 Sixth SIGHAN Workshop on Chinese Language Processing constxypL k k i ii +=  2 2 )()( 2)|(log)(   Malouf (2002) compared iterative procedures such as Generalized Iterative Scaling (GIS) and Improved Iterative Scaling (IIS) with numerical optimization techniques like limited-memory BFGS (L-BFGS) for estimating the maximum entropy model parameters and found that L-BFGS outperforms the other methods.
---------------------------------------------------
D08-1069:12	34:231	Model parameters were estimated with the limited memory variable metric algorithm and Gaussian smoothing (2=1000), using TADM (Malouf, 2002).
---------------------------------------------------
N07-1030:13	41:194	Specifically, we estimate parameters with the limited memory variable metric algorithm implemented in the Toolkit for Advanced Discriminative Modeling1 (Malouf, 2002).
---------------------------------------------------
E09-1048:14	137:218	3.3 Parameter Estimation There are various optimization methods that allow one to estimate the weights of features, including generalized iterative scaling and quasi-Newton methods (Malouf, 2002).
---------------------------------------------------
C04-1081:15	54:178	However, our implementation uses a quasi-Newton gradient-climber BFGS for optimization, which has been shown to converge much faster (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
P04-2009:16	34:110	The expanded set of results are summarised in Table 1, for Transformation Based Learning (TBL) (Brill, 1995), GIS based Maximum Entropy Modelling (GIS-MaxEnt) (Ratnaparkhi, 1998), L-BFGS based Maximum Entropy Modelling (L-BFGS-MaxEnt)2 (Malouf, 2002), Decision Tree Learning (Quinlan, 1993) and Memory Based Learning (MBL) (Daelemans et al. , 2002).
---------------------------------------------------
W03-1018:17	44:182	Malouf (2002) compares several algorithms for the ME estimation including GIS, IIS, and the limitedmemory variable metric (LMVM) method, which is a gradient-based method, and shows that the LMVM method requires much less time to converge for real NLP datasets.
---------------------------------------------------
C08-1106:18	124:203	As for numerical optimization (Malouf 2002; Wallach 2002), we performed gradient decent with the Limited-Memory BFGS (L-BFGS) optimization technique (Nocedal & Wright 1999).
---------------------------------------------------
P05-1002:19	40:188	In recent empirical studies on maximum entropy models and CRFs, limited memory variable metric (LMVM) has proven to be the most efficient method (Malouf, 2002; Wallach, 2002); accordingly, we have used LMVM for CRF estimation.
---------------------------------------------------
W03-0806:20	49:162	An example of this is the estimation of maximum entropy models, from simple iterative estimation algorithms used by Ratnaparkhi (1998) that converge very slowly, to complex techniques from the optimisation literature that converge much more rapidly (Malouf, 2002).
---------------------------------------------------
D08-1108:21	135:195	The optimal weight vector vector is obtained by maximizing the regularized log-likelihood (LLR), that is, vector = arg max vector LLR(vector) (4) To maximize the above function, we use a limitedmemory variable method (Benson and More, 2002) that is implemented in the TAO package (Benson et al., 2002) and has been shown to be very effective in various natural language processing tasks (Malouf, 2002).
---------------------------------------------------
H05-1084:22	61:183	The second learner, Maloufs probabilistic maximum entropy (Maxent) system (Malouf, 2002), uses the LMVM algorithm to estimate log-linear models.
---------------------------------------------------
W03-0430:23	41:87	It has recently been shown that quasi-Newton methods, such as L-BFGS, are significantly more efficient than traditional iterative scaling and even conjugate gradient (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
P07-1079:24	37:260	Given the HPSG treebank as training data, the model parametersi are estimated so as to maximize the log-likelihood of the training data (Malouf, 2002).
---------------------------------------------------
P07-1020:25	120:189	We found this algorithm to converge faster than the current state-ofthe-art in Maxent training, which is L2-regularized L-BFGS (Malouf, 2002)1.
---------------------------------------------------
P06-1066:26	108:243	Under the MaxEnt model, we have  = p(o|A1,A2) = exp( summationtext i ihi(o,A1,A2))summationtext o exp( summationtext i ihi(o,A1,A2))(10) where the functions hi {0,1}are model features and the i are weights of the model features which can be trained by different algorithms (Malouf, 2002).
---------------------------------------------------
W03-0403:27	71:178	The parse with the highest probability is taken as the preferred parse for the model.2 We use the limited memory variable metric algorithm (Malouf, 2002) to determine the weights.
---------------------------------------------------
J07-3005:28	333:569	Finally, with MaxEnt we use 40 iterations of the default L-BFGS parameter estimation (Malouf 2002).
---------------------------------------------------
I08-2078:29	79:119	In particular, the CRF trained by maximumlikelihood uses the L-BFGS algorithm (Liu and Nocedal, 1989), which converges quickly and gives a good performance on maximum entropy models (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
C04-1157:30	34:133	The expanded set of results are summarised in Table 1, for Transformation Based Learning (TBL) (Brill, 1995), GIS based Maximum Entropy Modelling (GIS-MaxEnt)2 (Ratnaparkhi, 1998), L-BFGS based Maximum Entropy Modelling (L-BFGSMaxEnt)3 (Malouf, 2002), Decision Tree Learning (Quinlan, 1993) and Memory Based Learning (MBL) (Daelemans et al. , 2002).
---------------------------------------------------
P07-1104:31	45:188	In our experiments, we used the limited memory quasi-Newton algorithm (or L-BFGS, Nocedal and Wright 1999) to find the optimal  because this method has been shown to be substantially faster than other methods such as Generalized Iterative Scaling (Malouf 2002).
---------------------------------------------------
N04-1042:32	40:128	Traditional maximum entropy learning algorithms, such as GIS and IIS (Pietra et al. , 1995), can be used to train CRFs, however, it has been found that a quasi-Newton gradient-climber, BFGS, converges much faster (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
P06-1088:33	33:187	We use the L-BFGS optimisation algorithm (Nocedal and Wright, 1999; Malouf, 2002) to perform the estimation.
---------------------------------------------------
W06-1661:34	7:278	Given a meaning representation in the form of Minimal Recursion Semantics (MRS; Copestake, Flickinger, Malouf, Riehemann, & Sag, 1995), the generator outputs English realizations in accordance with the HPSG LinGO English Resource Grammar (ERG; Flickinger, 2002).
---------------------------------------------------
W06-1661:35	58:278	Let us here analogously define CG D2 BP CUDC BD BNBMBMBMBNDC C9 CV to be the negative counterpart, so that for a given pair DC BP B4D7 CX BND6 CY B5 BE CG D2, we have that D6 CY BE CHB4D7 CX B5 but D6 CY is not annotated as a preferred realization of D7 CX . Fol1We use the TADM open-source package (Malouf, 2002) for training the models, using its limited-memory variable metric as the optimization method and experimentally determine the optimal convergence threshold and variance of the prior.
---------------------------------------------------
J05-1003:36	143:603	Collins and Koo Discriminative Reranking for NLP Della Pietra 1996; Della Pietra, Della Pietra, and Lafferty 1997), or conjugate gradient methods (Malouf 2002).
---------------------------------------------------
N03-1028:37	52:169	Those methods are very simple and guaranteed to converge, but as Minka (2001) and Malouf (2002) showed for classi cation, their convergence is much slower than that of general-purpose convex optimization algorithms when many correlated features are involved.
---------------------------------------------------
N03-1028:38	75:169	An earlier study indicates that L-BFGS performs well in maximum-entropy classi er training (Malouf, 2002).
---------------------------------------------------
N03-1028:39	133:169	The relative slowness of iterative scaling is also documented in a recent evaluation of training methods for maximum-entropy classi cation (Malouf, 2002).
---------------------------------------------------
P08-1024:40	88:227	We use a zero-mean Gaussian prior with the probability density function p0(k)  expparenleftbig2k/22parenrightbig.2 This results in the following log-likelihood objective and corresponding gradient: L= summationdisplay (e,f)D logp(e|f) + summationdisplay k logp0(k) (4) L k = Ep(d|e,f)[hk]Ep(e|f)[hk] k 2 (5) In order to train the model, we maximise equation (4) using L-BFGS (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
P09-2072:41	125:139	We include LBFGS as Malouf (2002) reported that it is better than other approaches including GIS 0 500 1000 1500 200010 2 101 100 101 Training Time (s) Relative function value difference   CD SCGIS GIS LBFGS (a) BROWN 0 50 100 150 20010 2 101 100 101 102 Training Time (s) Relative function value difference   CD SCGIS GIS LBFGS (b) CoNLL2000 0 500 1000 1500 200094 94.5 95 95.5 96 96.5 97 Training Time (s) Testing Accuracy   CD SCGIS GIS LBFGS (c) BROWN 0 50 100 150 20090 90.5 91 91.5 92 92.5 93 93.5 Training Time (s) F1 measure   CD SCGIS GIS LBFGS (d) CoNLL2000 Figure 2: First row: time versus the relative function value difference (17).
---------------------------------------------------
I08-1066:42	46:163	 = p (ojA1;A2) = exp( P i  ihi(o;A1;A2))P o exp( P i  ihi(o;A1;A2))(6) where the functions hi 2 f0;1g are model features and  i are weights of the model features trained automatically (Malouf, 2002).
---------------------------------------------------
P06-1009:43	55:213	We use L-BFGS, an iterative quasi-Newton optimisation method, which performs well for training log-linear models (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
W06-1619:44	48:171	Intuitively, the probability is defined as the normalized product of the weights exp(u) when a characteristic corresponding to fu appears in parse result T. The model parameters, u, are estimated using numerical optimization methods (Malouf, 2002) to maximize the log-likelihood of the training data.
---------------------------------------------------
W05-0612:45	134:219	We use the limited memory variable metric method as implemented in Maloufs maximum entropy package (2002) to set our weights.
---------------------------------------------------
W04-0817:46	51:146	We used the estimate software for estimation, which implements the LMVM algorithm (Malouf, 2002) and was kindly provided by Rob Malouf.
---------------------------------------------------
W07-1204:47	228:268	Specifically, we use the open-source Toolkit for Advanced Discriminative Modeling (TADM:2 Malouf, 2002) for training, using its limited-memory variable metric as the optimization method and determining best-performing convergence thresholds and prior sizes experimentally.
---------------------------------------------------
W02-2019:48	1:62	Markov models for language-independent named entity recognition Robert Malouf Alfa-Informatica Rijksuniversiteit Groningen Postbus 716 9700AS Groningen The Netherlands a0a2a1a4a3a6a5a8a7a4a9a11a10a12a3a14a13a14a15a17a16a19a18a14a7a4a20a17a16a22a21a23a3 1 Introduction This report describes the application of Markov models to the problem of language-independent named entity recognition for the CoNLL-2002 shared task (Tjong Kim Sang, 2002).
---------------------------------------------------
W07-1217:49	114:230	A similar approach has also been reported in (Johnson et al. , 1999; Riezler et al. , 2002; Malouf and van Noord, 2004).
---------------------------------------------------
W07-1217:50	119:230	The parameters can be efficiently estimated from a treebank, as shown by (Malouf, 2002).
---------------------------------------------------
W07-2208:51	45:192	Intuitively, the probability is defined as the normalized product of the weights exp(u) when a characteristic corresponding to fu appears in parse result T. The model parameters, u, are estimated using numerical optimization methods (Malouf, 2002) to maximize the log-likelihood of the training data.
---------------------------------------------------
W07-2208:52	11:192	Following this discriminative approach, techniques for efficiency were investigated for estimation (Geman and Johnson, 2002; Miyao and Tsujii, 2002; Malouf and van Noord, 2004) and parsing (Clark and Curran, 2004b; Clark and Curran, 2004a; Ninomiya et al. , 2005).
---------------------------------------------------
W06-1620:53	90:273	We use L-BFGS, an iterative quasi-Newton optimisation method, which performs well for training log-linear models (Malouf, 2002; Sha and Pereira, 2003).
---------------------------------------------------
P05-1011:54	21:172	Previous studies on probabilistic models for HPSG (Toutanova and Manning, 2002; Baldridge and Osborne, 2003; Malouf and van Noord, 2004) also adopted log-linear models.
---------------------------------------------------
P05-1011:55	28:172	Model parameters that maximize the loglikelihood of the training data are computed using a numerical optimization method (Malouf, 2002).
---------------------------------------------------
W03-1013:56	111:181	Note that the inside-outside approach can be combined with any maximum entropy estimation procedure, such as those evaluated by Malouf (2002).
---------------------------------------------------
W06-1206:57	166:193	The parameters < 1,2, > can be evaluated by maximising the pseudo-likelihood on a training corpus (Malouf, 2002).
---------------------------------------------------
W06-1206:58	164:193	With the efficient parameter estimation algorithms discussed by Malouf (2002), the training of the model is now very fast.
---------------------------------------------------
P05-1022:59	162:180	We used the Limited Memory Variable Metric optimization algorithm from the PETSc/TAO optimization toolkit (Benson et al. , 2004) to find the optimal feature weights  because this method seems substantially faster than comparable methods (Malouf, 2002).
---------------------------------------------------
P06-1071:60	9:208	Recent improvements on the original incremental feature selection (IFS) algorithm, such as Malouf (2002) and Zhou et al.
---------------------------------------------------
P06-1071:61	7:208	1 Introduction Conditional Maximum Entropy (CME) modeling has received a great amount of attention within natural language processing community for the past decade (e.g. , Berger et al. , 1996; Reynar and Ratnaparkhi, 1997; Koeling, 2000; Malouf, 2002; Zhou et al. , 2003; Riezler and Vasserman, 2004).
---------------------------------------------------
W05-1511:62	53:178	Most of them were developed for exhaustive parsing, i.e., producing all parse results that are given by the grammar (Matsumoto et al. , 1983; Maxwell and Kaplan, 1993; van Noord, 1997; Kiefer et al. , 1999; Malouf et al. , 2000; Torisawa et al. , 2000; Oepen et al. , 2002; Penn and Munteanu, 2003).
---------------------------------------------------
W05-1511:63	39:178	The probability of parse result T assigned to given sentence w = w1,,,wn is p(T|w) = 1Z w exp parenleftBiggsummationdisplay i ifi(T) parenrightBigg Zw = summationdisplay T prime exp parenleftBiggsummationdisplay i ifi(Tprime) parenrightBigg, where i is a model parameter, and fi is a feature function that represents a characteristic of parse tree T. Intuitively, the probability is defined as the normalized product of the weights exp(i) when a characteristic corresponding to fi appears in parse result T. Model parameters i are estimated using numer104 ical optimization methods (Malouf, 2002) so as to maximize the log-likelihood of the training data.
---------------------------------------------------
P05-1003:64	27:159	Malouf (2002) and Sha and Pereira (2003) show that gradient-based algorithms, particularly limited memory variable metric (LMVM), require much less time to reach convergence, for some NLP tasks, than the iterative scaling methods (Della Pietra et al. , 1997) previously used for log-linear optimisation problems.
---------------------------------------------------
E09-3005:65	123:229	For parameter estimation of the disambiguation model, in all reported experiments we use the TADM2 toolkit (toolkit for advanced discriminative training), with a Gaussian prior (2=1000) and the (default) limited memory variable metric estimation technique (Malouf, 2002).
---------------------------------------------------
W07-2207:66	4:381	Research groups working within grammatical frameworks like CCG (Clark & Curran, 2004), LFG (Riezler et al. , 2002), and HPSG (Malouf & van Noord, 2004; Oepen, Flickinger, Toutanova, & Manning, 2004; Miyao, Ninomiya, & Tsujii, 2005) have successfully integrated broad-coverage computational grammars with sophisticated statistical parse selection models.
---------------------------------------------------
W07-2207:67	192:381	A balance point between efficiency 5The models were trained using the open-source TADM package (Malouf, 2002), using default hyper-parameters for all configurations, viz.
---------------------------------------------------
W09-1204:68	106:132	For parameter estimation, we use the open source TADM system (Malouf, 2002).
---------------------------------------------------
W03-0407:69	55:172	A disadvantage is that the training times for ME models are usually relatively slow, especially with iterative scaling methods (see Malouf (2002) for alternative methods).
---------------------------------------------------
P09-2081:70	41:92	We use the L-BFGS algorithm (Malouf, 2002) and the Information Gain (IG) measure for parameter estimation and feature selection, respectively.
---------------------------------------------------
J05-4003:71	102:416	The parameter values that maximize the likelihood of a given training corpus can be computed using various optimization algorithms (see [Malouf 2002] for a comparison of such algorithms).
---------------------------------------------------
