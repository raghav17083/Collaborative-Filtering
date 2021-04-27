E99-1024:1	41:195	For that problem, some statistical methods have been applied and succeeded(Golding, 1995; Golding and Schabes, 1996).
---------------------------------------------------
P98-1003:2	178:204	Golding and Schabes (1996) propose a hybrid method that combines part-of-speech trigrams and context features in order to detect and correct realword errors.
---------------------------------------------------
W02-1005:3	53:132	When a90 is a word, 2Golding and Schabes (1996) show that the most important words for CSSC are contained within a window of a160a147a161 . 3The results shown were obtained for a162a163a59a165a164 with term weights doubled within a a160a147a161 context window.
---------------------------------------------------
W02-1005:4	118:132	Adjectives Nouns Verbs Overall Most Likely 63.43 66.52 57.6 63.09 Nave Bayes (FE) 75.67 84.15 76.65 80.16 Mixture 76.45 81.57 75.9 78.79 AdaMixt 76.83 83.39 77.10 80.16 MMVC 78.49 84.79 76.81 81.06 Table 3: Results using 5-fold cross validation on SENSEVAL1 training data (English) 5.3 Spelling Correction Both MM and the enhanced Bayes model obtain virtually the same overall performance9 as the TriBayes system reported in (Golding and Schabes, 1996), which uses a similar feature space.
---------------------------------------------------
A97-1025:5	123:209	Golding and Schabes (1996) have already shown that using a trigram model to predict words from a confusion set based on the expected part of speech is very effective.
---------------------------------------------------
A97-1025:6	153:209	Table 2 also gives the results obtained by Tribayes as reported in (Golding and Schabes, 1996).
---------------------------------------------------
A97-1025:7	133:209	Consequently, our sentence counts for the various confusion sets differ slightly from the counts reported in (Golding and Schabes, 1996).
---------------------------------------------------
A97-1025:8	30:209	Recently, Golding and Schabes (1996) described a system, Tribayes, that combines a trigram model of the words' parts of speech with a Bayesian classifier.
---------------------------------------------------
A97-1025:9	174:209	The results of Tribayes (Golding and Schabes, 1996) are also given.
---------------------------------------------------
A97-1025:10	162:209	The baseline predictor presented in this paper and in (Golding and Schabes, 1996) are based on the same method so the correspond170 ing columns in Table 2 can be compared to get an idea of the distribution of sentences that contain the most frequent word for each confusion set.
---------------------------------------------------
P97-1067:11	14:71	2 Generalizing Lexical Co-occurrence 2.1 Evidence-based Models of Context Evidence-based models represent context as a set of features, say words, that are observed to co-occur with, and thereby predict, a word (Yarowsky, 1992; Golding and Schabes, 1996; Karow and Edelman, 1996; Ng and Lee, 1996).
---------------------------------------------------
A00-3005:12	33:138	 Error patterns (Kukich 1992; Golding and Schabes 1996; Mangu and Brill 1997), in the form of statistical information, hand-coded rules or automatically learned ones.
---------------------------------------------------
D09-1093:13	51:305	This has led to the two sub-tasks being approached separately (Golding and Schabes, 1996).
---------------------------------------------------
D07-1021:14	16:254	1 The use of contextual language models in spelling correction has been discussed elsewhere: (Church and Gale, 1991), (Mays et al, 1991), (Kukich, 1992) and (Golding and Schabes, 1996).
---------------------------------------------------
P01-1005:15	19:133	The more recent set of techniques includes mult iplicative weightupdate algorithms (Golding and Roth, 1998), latent semantic analysis (Jones and Martin, 1997), transformation-based learning (Mangu and Brill, 1997), differential grammars (Powers, 1997), decision lists (Yarowsky, 1994), and a variety of Bayesian classifiers (Gale et al. , 1993, Golding, 1995, Golding and Schabes, 1996).
---------------------------------------------------
P98-2152:16	17:203	Recently, statistical language models and featurebased method have been used for context-sensitive spelling correction, where errors are corrected considering the context in which the error occurs (Church and Gale, 1991; Mays et al. , 1991; Golding and Schabes, 1996).
---------------------------------------------------
W97-0117:17	145:167	Overall, the C-Box approach is partly related to error-driven learning techniques as used for partof-speech tagging (Bfill, 1992, 1995), and spelling correction (e.g. , Golding & Schabes, 1996).
---------------------------------------------------
C08-2008:18	102:120	Regarding their treatment, there have been proposals ranging from error patterns (Kukich 1992; Golding and Schabes 1996), in the form of handcoded rules or automatically learned ones, to systems that integrate syntactic analysis.
---------------------------------------------------
N04-1016:19	111:228	We used the same test set (2056 tokens from the Brown corpus) and confusion sets as Golding and Schabes (1996), Mangu and Brill (1997), and Cucerzan and Yarowsky (2002).
---------------------------------------------------
N04-1016:20	103:228	Most methods are trained and tested on Model Alta BNC Model Alta BNC f (t) 72.98 70.00 f (w1;t;w2)= f (t) 87.77 76.33 f (w1;t) 84.40 83.02 f (w1;w2;t)= f (t) 86.27 74.47 f (t;w1) 84.89 82.74 f (t;w2;w2)= f (t) 84.94 74.23 f (w1;t;w2) 89.24#*77.13 f (w1;t;w2)= f (w1;t) 80.70 73.69 f (w1;w2;t) 87.13 74.89 f (w1;t;w2)= f (t;w2) 82.24 75.10 f (t;w1;w2) 84.68 75.08 f (w1;w2;t)= f (w2;t) 72.11 69.28 f (w1;t)= f (t) 82.81 77.84 f (t;w1;w2)= f (t;w1) 75.65 72.57 f (t;w1)= f (t) 77.49 80.71# Table 5: Performance of Altavista counts and BNC counts for context sensitive spelling correction (data from Cucerzan and Yarowsky 2002) Model Accuracy Baseline BNC 70.00 Baseline Altavista 72.98 Best BNC 80.71 Golding (1995) 81.40 Jones and Martin (1997) 84.26 Best Altavista 89.24 Golding and Schabes (1996) 89.82 Mangu and Brill (1997) 92.79 Cucerzan and Yarowsky (2002) 92.20 Golding and Roth (1999) 94.23 Table 6: Performance comparison with the literature for context sensitive spelling correction the Brown corpus, using 80% for training and 20% for testing.3 We devised a simple, unsupervised method for performing spelling correction using web counts.
---------------------------------------------------
N04-1016:21	98:228	These include a variety of Bayesian classifiers (Golding, 1995; Golding and Schabes, 1996), decision lists (Golding, 1995) transformation-based learning (Mangu and Brill, 1997), Latent Semantic Analysis (LSA) (Jones and Martin, 1997), multiplicative weight update algorithms (Golding and Roth, 1999), and augmented mixture models (Cucerzan and Yarowsky, 2002).
---------------------------------------------------
N04-1016:22	116:228	A comparison with the literature shows that the best Altavista model outperforms Golding (1995), Jones and Martin (1997) and performs similar to Golding and Schabes (1996).
---------------------------------------------------
