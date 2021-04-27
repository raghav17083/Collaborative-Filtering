W08-1105:1	25:178	Most approaches are supervised and require training data to learn which words or constituents can be dropped from a sentence (Riezler et al., 2003; McDonald, 2006).
---------------------------------------------------
P09-2026:2	29:88	This score is factored using a first-order Markov assumption over the words in the compressed sentence, and is defined by the dot product between a high dimensional feature representation and a corresponding weight vector (for details, refer to McDonald, 2006).
---------------------------------------------------
P09-2026:3	15:88	McDonalds (2006) model (M06, henceforth) is similar to K&M except that it uses discriminative online learning to train feature weights.
---------------------------------------------------
D07-1008:4	199:382	These included gold standard compressions and the output of our system and McDonalds (2006).
---------------------------------------------------
D07-1008:5	46:382	McDonald (2006) also presents a sentence compression model that uses a discriminative large margin algorithm.
---------------------------------------------------
D07-1008:6	270:382	We are grateful to James Clarke for sharing his implementation of McDonald (2006) with us.
---------------------------------------------------
D07-1008:7	197:382	Table 1: Compression examples from the Broadcast news corpus (O: original sentence, M: McDonald (2006), S: STSG, G: gold standard) Ziff-Davis and one for the Broadcast news dataset.
---------------------------------------------------
D07-1008:8	259:382	We empirically evaluate our approach against a state-of-the art model (McDonald, 2006) and show performance gains on two compression corpora.
---------------------------------------------------
D07-1008:9	266:382	Nevertheless, improvements should be possible by incorporating features defined over n-grams and dependencies (McDonald, 2006).
---------------------------------------------------
D07-1008:10	15:382	A notable exception is sentence compression for which end-to-end rewriting systems are commonly developed (Knight and Marcu, 2002; Turner and Charniak, 2005; Galley and McKeown, 2007; Riezler et al. , 2003; McDonald, 2006).
---------------------------------------------------
D07-1008:11	184:382	We used an implementation of McDonald (2006)forcomparisonofresults(ClarkeandLapata, 2007).
---------------------------------------------------
D07-1008:12	167:382	We give details of the corpora used, briefly introduce McDonalds (2006) sentence compression model used for comparison with our approach, and explain how system output was evaluated.
---------------------------------------------------
D07-1008:13	177:382	Comparison with State-of-the-art We evaluated our approach against McDonalds (2006) discriminativemodel.Thismodelisagoodbasisforcomparison for several reasons.
---------------------------------------------------
D07-1008:14	231:382	Table 2 shows the average compression rates (CompR) for McDonald (2006)andourmodel(STSG)aswellastheirperformance according to grammatical relations F1.
---------------------------------------------------
P08-1035:15	6:147	Despite its limited scale, prior work in sentence compression relied heavily on this particular corpus for establishing results (Turner and Charniak, 2005; McDonald, 2006; Clarke and Lapata, 2006; Galley and McKeown, 2007).
---------------------------------------------------
P08-1035:16	4:147	1 Introduction For better or worse, much of prior work on sentence compression (Riezler et al., 2003; McDonald, 2006; Turner and Charniak, 2005) turned to a single corpus developed by Knight and Marcu (2002) (K&M, henceforth) for evaluating their approaches.
---------------------------------------------------
P08-1035:17	13:147	Thus, unlike McDonald (2006), Clarke and Lapata (2006) and Cohn and Lapata (2007), we do not insist on nding a globally optimal solution in the space of 2n possible compressions for an n word long sentence.
---------------------------------------------------
W09-1801:18	169:244	The best results were obtained without iterative shortening, which is explained by the fact that the selection of the lowest leftmost S node (first tive model described by McDonald (2006), which captures soft syntactic evidence (we reproduced the same set of features).
---------------------------------------------------
W09-1801:19	191:244	We can see that both variants outperform the Hedge Trimmer baseline by a great margin, and are in line with the system of McDonald (2006); however, none of our variants employ a phrase-structure parser.
---------------------------------------------------
W09-1801:20	86:244	Non-localinformation,such as arity (or valency) and neighbouringdependencies,can be crucial to obtaininghigh parsingaccuracies(Kleinand Manning, 2002; McDonaldand Pereira, 2006).
---------------------------------------------------
W09-1801:21	152:244	Ratio P R F1 P R F1 HedgeTrimmer 57.64% 0.7099 0.5925 0.6459 0.7195 0.6547 0.6367 McDonald (2006) 71.40% 0.7444 0.7697 0.7568 0.7711 0.7852 0.7696 NoBigram 71.20% 0.7399 0.7626 0.7510 0.7645 0.7730 0.7604 Bigram 71.35% 0.7472 0.7720 0.7594 0.7737 0.7848 0.7710 Table 1: Results for sentence compression in the Clarkes test dataset (441 sentences) for our implementation of the baseline systems (HedgeTrimmer and the system described in McDonald, 2006), and the two variants of our model, NoBigram and Bigram.
---------------------------------------------------
W09-1801:22	98:244	Recentlytherehave alsobeenproposalsfor exhaustive methodsthatweaken theedge-factoredassumption,includingbothapproximatemethods(McDonaldandPereira,2006)andexactmethodsthroughinteger linearprogramming(RiedelandClarke, 2006) or branch-and-boundalgorithms(Hirakawa, 2006).
---------------------------------------------------
W09-1801:23	93:244	For neighbouring parse decisions,we extend the work of McDonald and Pereira(2006)and show that modelingvertical neighbourhoodsmakes parsingintractablein additionto modelinghorizontalneighbourhoods.A consequenceof these results is that it is unlikely that exactnon-projective dependency parsingis tractable for any model assumptionsweaker thanthosemade by the edge-factoredmodels.
---------------------------------------------------
W09-1801:24	179:244	As in McDonald (2006), we include features that depend on the in-between words in the original sentence that are to be omitted in the compression.11 As stated in 3.2, inference through this model can be done by solving an ILP with O(N2) variables and constraints.
---------------------------------------------------
W09-1801:25	61:244	3.1 Related Work Past approaches to sentence compression include a noisy channel formulation (Knight and Marcu, 2000; Daume and Marcu, 2002), heuristic methods that parse the sentence and then trim constituents according to linguistic criteria (Dorr et al., 2003; Zajic et al., 2006), a pure discriminative model (McDonald, 2006), and an ILP formulation (Clarke and Lapata, 2008).
---------------------------------------------------
W09-1801:26	63:244	McDonald (2006) uses the outputs of two parsers (a phrase-based and a dependency parser) as features in a discriminative model that decomposes over pairs of consecutive words.
---------------------------------------------------
W09-1801:27	182:244	html 10http://sourceforge.net/projects/mstparser 11The major difference between this variant and model of McDonald (2006) is that the latter employs soft syntactic evidence as input features, while we make the dependency relations part of the output features.
---------------------------------------------------
P08-2035:28	87:92	Naturally, discriminative models such as McDonald (2006) are also likely to improve by using the added data.
---------------------------------------------------
P09-1093:29	12:165	An alternative to the tree trimming approach is the sequence-oriented approach (McDonald, 2006; Nomoto, 2007; Clarke and Lapata, 2006; Hori and Furui, 2003).
---------------------------------------------------
P09-1093:30	151:165	Nomoto (2007) and McDonald (2006) employed the random field based approach.
---------------------------------------------------
P09-1093:31	150:165	As an alternative to these tree trimming approaches, sequence-oriented approaches have been proposed (McDonald, 2006; Nomoto, 2007; Hori and Furui, 2003; Clarke and Lapata, 2006).
---------------------------------------------------
W09-1802:32	203:208	Better statistical methods have been developed for producing high quality compression candidates (McDonald, 2006), that maintain linguistic quality, some recent work even uses ILPs for exact inference (Clarke and Lapata, 2008).
---------------------------------------------------
W06-2932:33	12:130	This has been shown through their successful use in many standard natural language processing tasks, including machine translation (Ding and Palmer, 2005), sentence compression (McDonald, 2006), and textual inference (Haghighi et al. , 2005).
---------------------------------------------------
W06-2932:34	71:130	These results show that the discriminative spanning tree parsing framework (McDonald et al. , 2005b; McDonald and Pereira, 2006) is easily adapted across all these languages.
---------------------------------------------------
W06-2932:35	124:130	(McDonald et al. , 2005b; McDonald and Pereira, 2006) generalizes well to languages other than English.
---------------------------------------------------
W06-2932:36	20:130	2 Stage 1: Unlabeled Parsing The first stage of our system creates an unlabeled parse y for an input sentence x. This system is primarily based on the parsing models described by McDonald and Pereira (2006).
---------------------------------------------------
W06-2932:37	45:130	Furthermore, it made the system homogeneous in terms of learning algorithms since that is what is used to train our unlabeled parser (McDonald and Pereira, 2006).
---------------------------------------------------
W06-2932:38	3:130	The first stage is based on the unlabeled dependency parsing models described by McDonald and Pereira (2006) augmented with morphological features for a subset of the languages.
---------------------------------------------------
P06-2019:39	14:240	In a discriminative setting (Knight and Marcu 2002; Riezler et al. 2003; McDonald 2006), sentences are represented by a rich feature space (typically induced from parse trees) and the goal is to learn rewrite rules indicating which words should be deleted in a given context.
---------------------------------------------------
P06-2019:40	62:240	In other models (Hori and Furui 2004; McDonald 2006) the compression score is maximised 145 using dynamic programming.
---------------------------------------------------
P06-2019:41	41:240	Discriminative formulations of the compression task include decision-tree learning (Knight and Marcu 2002), maximum entropy (Riezler et al. 2003), support vector machines (Nguyen et al. 2004), and large-margin learning (McDonald 2006).
---------------------------------------------------
P06-2019:42	166:240	Finally, the model that includes the signi cance score was optimised against a loss function similar to McDonald (2006) to bring the language model and the score into harmony.
---------------------------------------------------
P06-2019:43	179:240	To counteract this, human judgements are often collected on compression output; however the evaluations are limited to small subject pools (often four judges; Knight and Marcu 2002; Turner and Charniak 2005; McDonald 2006) which makes dif cult to apply inferential statistics on the data.
---------------------------------------------------
C08-1018:44	15:275	Indeed, a variety of models have been successfully developed for this task ranging from instantiations of the noisy-channel model (Knight and Marcu, 2002; Galley and McKeown, 2007; Turner and Charniak, 2005), to large-margin learning (McDonald, 2006; Cohn and Lapata, 2007), and Integer Linear Programming (Clarke, 2008).
---------------------------------------------------
C08-1018:45	13:275	The simplification renders the task computationally feasible, allowing efficient decoding using a dynamic program (Knight and Marcu, 2002; Turner and Charniak, 2005; McDonald, 2006).
---------------------------------------------------
C08-1018:46	209:275	Note that their model is a strong baseline: it performed significantly better than competitive approaches (McDonald, 2006) across a variety of compression corpora.
---------------------------------------------------
D09-1041:47	5:178	1 Introduction Whilethereareafewnotableexceptions(Horiand Furui, 2004; Yamagata et al., 2006), it would be safe to say that much of prior research on sentence compression has been focusing on what we might call model-intensive approaches, where the goal is to mimic human created compressions as faithfully as possible, using probabilistic and/or machine learning techniques (Knight and Marcu, 2002; Riezler et al., 2003; Turner and Charniak, 2005; McDonald, 2006; Clarke and Lapata, 2006; Cohn and Lapata, 2007; Cohn and Lapata, 2008; Cohn and Lapata, 2009).
---------------------------------------------------
P06-2109:48	53:230	McDonald (McDonald, 2006) independently proposed a new machine learning approach.
---------------------------------------------------
P06-1048:49	17:244	Many algorithms exploit parallel corpora (Jing 2000; Knight and Marcu 2002; Riezler et al. 2003; Nguyen et al. 2004a; Turner and Charniak 2005; McDonald 2006) to learn the correspondences between long and short sentences in a supervised manner, typically using a rich feature space induced from parse trees.
---------------------------------------------------
P06-1048:50	36:244	It has inspired many discriminative approaches to the compression task (Riezler et al. 2003; Nguyen et al. 2004b; McDonald 2006) and has been extended to languages other than English (see Nguyen et al. 2004a).
---------------------------------------------------
P06-1048:51	102:244	Ziff-Davis Corpus Most previous work (Jing 2000; Knight and Marcu 2002; Riezler et al. 2003; Nguyen et al. 2004a; Turner and Charniak 2005; McDonald 2006) has relied on automatically constructed parallel corpora for training and evaluation purposes.
---------------------------------------------------
E09-1017:52	12:229	Considerations of sentence fluency are also key in sentence simplification (Siddharthan, 2003), sentence compression (Jing, 2000; Knight and Marcu, 2002; Clarke and Lapata, 2006; McDonald, 2006; Turner and Charniak, 2005; Galley and McKeown, 2007), text re-generation for summarization (Daume III and Marcu, 2004; Barzilay and McKeown, 2005; Wan et al., 2005) and headline generation (Banko et al., 2000; Zajic et al., 2007; Soricut and Marcu, 2007).
---------------------------------------------------
N07-1023:53	145:175	One successful recent approach (McDonald, 2006) combines a discriminative framework with a set of features that capture information similar to the K&M model.
---------------------------------------------------
