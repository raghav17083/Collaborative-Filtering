W06-1652:1	100:189	3 Data Sets We used three opinion-related data sets for our analyses and experiments: the OP data set created by (Wiebe et al. , 2004), the Polarity data set5 created by (Pang and Lee, 2004), and the MPQA data set created by (Wiebe et al. , 2005).6 The OP and Polarity data sets involve document-level opinion classi cation, while the MPQA data set involves 5Version v2.0, which is available at: http://www.cs.cornell.edu/people/pabo/movie-review-data/ 6Available at http://www.cs.pitt.edu/mpqa/databaserelease/ sentence-level classi cation.
---------------------------------------------------
W06-1652:2	185:189	Lexical cues of differing complexities have been used, including single words and Ngrams (e.g. , (Mullen and Collier, 2004; Pang et al. , 2002; Turney, 2002; Yu and Hatzivassiloglou, 2003; Wiebe et al. , 2004)), as well as phrases and lexico-syntactic patterns (e.g, (Kim and Hovy, 2004; Hu and Liu, 2004; Popescu and Etzioni, 2005; Riloff and Wiebe, 2003; Whitelaw et al. , 2005)).
---------------------------------------------------
W05-0305:3	53:201	3 3 Annotation of attribution Wiebe and her colleagues have pointed out the importance of ascribing beliefs and assertions expressed in text to the agent(s) holding or making them (Riloff and Wiebe, 2003; Wiebe et al. , 2004; Wiebe et al. , 2005).
---------------------------------------------------
W05-0305:4	8:201	A preliminary report on this project was presented at the 2004 workshop on Frontiers in Corpus Annotation (Miltsakaki et al. , 2004a), where we described our annotation of discourse connectives (both explicit and implicit) along with their (clausal) arguments.
---------------------------------------------------
W05-0305:5	21:201	(2004a), Miltsakaki et al.
---------------------------------------------------
N06-3005:6	29:72	So far research in automatic opinion recognition has primarily addressed learning subjective language (Wiebe et al. , 2004; Riloff et al. , 2003; Riloff and Wiebe, 2003), identifying opinionated documents (Yu and Hatzivassiloglou, 2003) and sentences (Yu and Hatzivassiloglou, 2003; Riloff et al. , 2003; Riloff and Wiebe, 2003), and discriminating between positive and negative language (Yu and Hatzivassiloglou, 2003; Turney and Littman, 2003; Pang et al. , 2002; Dave et al. , 2003; Nasukawa and Yi, 2003; Morinaga et al. , 2002).
---------------------------------------------------
P07-1125:7	45:186	Other relevant recent work includes (Zhang, 2004), in which random feature projection and a committee of SVM classifiers is used in a hybrid co/self-training strategy for weakly supervised relation classification and (Chen et al. , 2006) where a graph based algorithm called label propagation is employed to perform weakly supervised relation extraction.
---------------------------------------------------
P07-1125:8	114:186	In the same vein, for the case of entity/relation extraction and classification (Collins and Singer, 1999; Zhang, 2004; Chen et al. , 2006) the context of the entity or entities in consideration provides a highly relevant feature space.
---------------------------------------------------
P07-1125:9	160:186	997 0.58 0.6 0.62 0.64 0.66 0.68 0.7 0.72 0.74 0.76 0.78 0.8 0 20 40 60 80 100 120 140 BEP Iteration Prob (Prob)Prob (SVM) SVM (Prob)SVM (SVM) Baseline Prob (Prob) denotes our probabilistic learning model and classifier (9) Prob (SVM) denotes probabilistic learning model with SVM classifier SVM (Prob) denotes committee-based model (10.4) with probabilistic classifier SVM (SVM) denotes committee-based model with SVM classifier Baseline denotes substring matching classifier of (Light et al. , 2004) Figure 1: Learning curves 10.4 Baselines As a baseline classifier we use the substring matching technique of (Light et al. , 2004), which labels a sentence as spec if it contains one or more of the following: suggest, potential, likely, may, at least, in part, possibl, further investigation, unlikely, putative, insights, point toward, promise and propose.
---------------------------------------------------
N07-1039:10	16:46	The use of multiple text classifiers by Wiebe and colleagues (Wilson et al. , 2005; Wiebe et al. , 2004) for various kinds of sentiment classification can also be viewed as a sentencelevel technique for analyzing appraisal expressions.
---------------------------------------------------
N07-1039:11	9:46	Sentiment analysis includes a variety of different problems, including: sentiment classification techniques to classify reviews as positive or negative, based on bag of words (Pang et al. , 2002) or positive and negative words (Turney, 2002; Mullen and Collier, 2004); classifying sentences in a document as either subjective or objective (Riloff and Wiebe, 2003; Pang and Lee, 2004); identifying or classifying appraisal targets (Nigam and Hurst, 2004); identifying the source of an opinion in a text (Choi et al. , 2005), whether the author is expressing the opinion, or whether he is attributing the opinion to someone else; and developing interactive and visual opinion mining methods (Gamon et al. , 2005; Popescu and Etzioni, 2005).
---------------------------------------------------
P09-1029:12	106:211	IR models, such as Vector Space (VS), probabilistic models such as BM25, and Language Modeling (LM), albeit in different forms of approach and measure, employ heuristics and formal modeling approaches to effectively evaluate the relevance of a term to a document (Fang et al., 2004).
---------------------------------------------------
P09-1029:13	26:211	Also, syntactic features such as the dependency relationship of words and subtrees have been shown to effectively improve the performances of sentiment analysis (Kudo and Matsumoto, 2004; Gamon, 2004; Matsumoto et al., 2005; Ng et al., 2006).
---------------------------------------------------
P09-1029:14	28:211	Accordingly, much research has focused on recognizing terms semantic orientations and strength, and compiling sentiment lexicons (Hatzivassiloglou and Mckeown, 1997; Turney and Littman, 2003; Kamps et al., 2004; Whitelaw et al., 2005; Esuli and Sebastiani, 2006).
---------------------------------------------------
P09-1029:15	45:211	The rare occurrence of terms in document collections has been regarded as a very important feature in IR methods, and effective IR models of today, either explicitly or implicitly, accommodate this feature as an Inverse Document Frequency (IDF) heuristic (Fang et al., 2004).
---------------------------------------------------
W06-0305:16	9:177	Recent work has shown the importance of recognizing such perspectivization of information for several NLP applications, such as information extraction, summarization, question answering (Wiebe et al. , 2004; Stoyanov et al. , 2005; Riloff et al. , 2005) and generation (Prasad et al. , 2005).
---------------------------------------------------
W06-0305:17	48:177	Most of the annotation approaches tackling these issues, however, are aimed at performing classifications at either the document level (Pang et al. , 2002; Turney, 2002), or the sentence or word level (Wiebe et al. , 2004; Yu and Hatzivassiloglou, 2003).
---------------------------------------------------
W06-0305:18	12:177	This paper describes an extended annotation scheme for marking the attribution of discourse relations and their arguments annotated in the Penn Discourse TreeBank (PDTB) (Miltsakaki et al. , 2004; Prasad et al. , 2004; Webber et al. , 2005), the primary goal being to capture the source and degrees of factuality of abstract objects.
---------------------------------------------------
D07-1113:19	53:290	These techniques were applied and examined in different domains, such as customer reviews (Hu and Liu 2004; Popescu et al. , 2005) and news articles (Kim and Hovy, 2004; Wilson et al. , 2005).
---------------------------------------------------
H05-1073:20	42:192	(Wiebe et al, 2004), measuring strength of subjective clauses (Wilson, Wiebe and Hwa, 2004), determining word polarity (Hatzivassiloglou and McKeown, 1997) or texts attitudinal valence, e.g.
---------------------------------------------------
H05-1073:21	34:192	A short study by (Sugimoto et al. , 2004) addresses sentence-level emotion recognition for Japanese TTS.
---------------------------------------------------
H05-1073:22	112:192	For the adjectives, Py-WordNets (Steele et al. , 2004) SIMILAR feature was used to retrieve similar items of the primary emotion adjectives, exploring one additional level in the hierarchy (i.e. similar items of all senses of all words in the synset).
---------------------------------------------------
P08-1033:23	168:191	Our finding that token unigram features are capable of solving the task accurately agrees with the the results of previous works on hedge classification ((Light et al., 2004), (Med287 lock and Briscoe, 2007)), and we argue that 2-3 word-long phrases also play an important role as hedge cues and as non-speculative uses of an otherwise speculative keyword as well (i.e. to resolve an ambiguity).
---------------------------------------------------
P08-1033:24	154:191	(Light et al., 2004) and Baseline 2 denotes the system of Medlock and Briscoe (Medlock and Briscoe, 2007).
---------------------------------------------------
P08-1033:25	170:191	((Wiebe et al., 2004)), who addressed the broader task of subjectivity learning and found that the density of other potentially subjective cues in the context benefits classification accuracy, we observed that the co-occurence of speculative cues in a sentence does not help in classifying a term as speculative or not.
---------------------------------------------------
P08-1033:26	27:191	This phenomenon, together with others used to express forms of authorial opinion, is often classified under the notion of subjectivity (Wiebe et al., 2004), (Shanahan et al., 2005).
---------------------------------------------------
P08-1033:27	107:191	3.1.3 Results obtained adding external dictionaries In our final model we added the keywords used in (Light et al., 2004) and those gathered for our ICD9-CM hedge detection module.
---------------------------------------------------
P08-1033:28	155:191	For clinical free texts, Baseline 1 is an out-domain model since the keywords were collected for scientific texts by (Light et al., 2004).
---------------------------------------------------
P08-1033:29	28:191	Previous studies (Light et al., 2004) showed that the detection of hedging can be solved effectively by looking for specific keywords which imply that the content of a sentence is speculative and constructing simple expert rules that describe the circumstances of where and how a keyword should appear.
---------------------------------------------------
W06-2915:30	38:149	Research on the automatic classification of movie or product reviews as positive or negative (e.g. , (Pang et al. , 2002; Morinaga et al. , 2002; Turney and Littman, 2003; Nasukawa and Yi, 2003; Mullen and Collier, 2004; Beineke et al. , 2004; Hu and Liu, 2004)) is perhaps the most similar to our work.
---------------------------------------------------
W06-2915:31	35:149	So far, research in automatic opinion recognition has primarily addressed learning subjective language (Wiebe et al. , 2004; Riloff et al. , 2003), identifying opinionated documents (Yu and Hatzivassiloglou, 2003) and sentences (Yu and Hatzivassiloglou, 2003; Riloff et al. , 2003), and discriminating between positive and negative language (Pang et al. , 2002; Morinaga et al. , 2002; Yu and Hatzivassiloglou, 2003; Turney and Littman, 2003; Dave et al. , 2003; Nasukawa and Yi, 2003; Popescu and Etzioni, 2005; Wilson et al. , 2005).
---------------------------------------------------
N09-1056:32	38:259	(Yu and Hatzivassiloglou, 2003; Wiebe et al., 2004; Finn and Kushmerick, 2006; Ni et al., 2007; Stepinski and Mittal, 2007).
---------------------------------------------------
N09-1056:33	52:259	Examples of such early work include (Turney, 2002; Pang et al., 2002; Dave et al., 2003; Hu and Liu, 2004; Popescu and Etzioni, 2005).
---------------------------------------------------
W09-1904:34	26:182	Previous research has focused on classifying subjective-versus-objective expressions (Wiebe et al., 2004), and also on accurate sentiment polarity assignment (Turney, 2002; Yi et al., 2003; Pang and Lee, 2004; Sindhwani and Melville, 2008; Melville et al., 2009).
---------------------------------------------------
P06-1133:35	45:268	There are also research work on automatically classifying movie or product reviews as positive or negative (Nasukawa and Yi, 2003; Mullen and Collier, 2004; Beineke et al. , 2004; Pang and Lee, 2004; Hu and Liu, 2004).
---------------------------------------------------
P06-1133:36	44:268	There are studies on learning subjective language (Wiebe et al. , 2004), identifying opinionated documents (Yu and Hatzivassiloglou, 2003) and sentences (Riloff et al. , 2003; Riloff and Wiebe, 2003), and discriminating between positive and negative language (Turney and Littman, 2003; Pang et al. , 2002; Dave et al. , 2003; Nasukawa and Yi, 2003; Morinaga et al. , 2002).
---------------------------------------------------
D08-1098:37	202:220	Also related is research on sentiment analysis (e.g., Pang et al., 2004) where the goal is to classify a sentence or text fragment as being overall positive or negative.
---------------------------------------------------
D08-1098:38	203:220	More generally, (Wiebe et al. 2004) and subsequent work focused on the analysis of subjective language in narrative text, primarily news.
---------------------------------------------------
P06-2079:39	48:254	Much work has been performed on learning to identify and classify polarity terms (i.e. , terms expressing a positive sentiment (e.g. , happy) or a negative sentiment (e.g. , terrible)) and exploiting them to do polarity classification (e.g. , Hatzivassiloglou and McKeown (1997), Turney (2002), Kim and Hovy (2004), Whitelaw et al.
---------------------------------------------------
P06-2079:40	34:254	(2003), Yu and Hatzivassiloglou (2003), Kim and Hovy (2004), Gamon et al.
---------------------------------------------------
P06-2079:41	7:254	(2004)) to opinion-oriented information extraction (e.g. , Riloff et al.
---------------------------------------------------
P06-2079:42	125:254	4.1 Experimental Setup Like several previous work (e.g. , Mullen and Collier (2004), Pang and Lee (2004), Whitelaw et al.
---------------------------------------------------
P06-2079:43	93:254	Admittedly, the high accuracy achieved using such a simple set of features is somewhat surprising, although it is consistent with previous results on document-level subjectivity classification in which accuracies of 94-97% were obtained (Yu and Hatzivassiloglou, 2003; Wiebe et al. , 2004).
---------------------------------------------------
