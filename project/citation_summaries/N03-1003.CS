P06-1114:1	118:248	Classi er Training Set Precision Recall F-Measure Linear 10K pairs 0.837 0.774 0.804 Maximum Entropy 10K pairs 0.881 0.851 0.866 Maximum Entropy 450K pairs 0.902 0.944 0.922 Table 4: Performance of Alignment Classi er 3.2 Paraphrase Acquisition Much recent work on automatic paraphrasing (Barzilay and Lee, 2003) has used relatively simple statistical techniques to identify text passages that contain the same information from parallel corpora.
---------------------------------------------------
P06-1114:2	123:248	In order increase the likelihood that 909 only true paraphrases were considered as phraselevel alternations for an example, extracted sentences were clustered using complete-link clustering using a technique proposed in (Barzilay and Lee, 2003).
---------------------------------------------------
P06-2027:3	126:242	The procedure of substituting named entities with their respective tags previously proved to be useful for various tasks (Barzilay and Lee, 2003; Sudo et al. , 2003; Filatova and Prager, 2005).
---------------------------------------------------
P06-2027:4	90:242	Many of the current approaches of domain modeling collapse together different instances and make the decision on what information is important for a domain based on this generalized corpus (Collier, 1998; Barzilay and Lee, 2003; Sudo et al. , 2003).
---------------------------------------------------
W07-0716:5	46:171	Most previous work on paraphrase has focused on high quality rather than coverage (Barzilay and Lee, 2003; Quirk et al. , 2004), but generating artificial references for MT parameter tuning in our setting has two unique properties compared to other paraphrase applications.
---------------------------------------------------
W07-0716:6	139:171	At the sentence level, (Barzilay and Lee, 2003) employed an unsupervised learning approach to cluster sentences and extract lattice pairs from comparable monolingual corpora.
---------------------------------------------------
P08-1089:7	63:250	Barzilay and Lee (2003) applied multi-sequence alignment (MSA) to parallel news sentences and induced paraphrase patterns for generating new sentences (Figure 1 (1)).
---------------------------------------------------
P08-1089:8	17:250	In paraphrase generation, a text unit that matches a pattern P can be rewritten using the paraphrase patterns of P. Avarietyofmethodshavebeenproposedonparaphrase patterns extraction (Lin and Pantel, 2001; Ravichandran and Hovy, 2002; Shinyama et al., 2002; Barzilay and Lee, 2003; Ibrahim et al., 2003; Pang et al., 2003; Szpektor et al., 2004).
---------------------------------------------------
P08-1089:9	164:250	Some methods only extract paraphrase patternsusingnewsarticlesoncertaintopics(Shinyama et al., 2002; Barzilay and Lee, 2003), while some others need seeds as initial input (Ravichandran and Hovy, 2002).
---------------------------------------------------
P08-1089:10	60:250	The preci781 start Palestinian suicide bomberblew himself up in SLOT1 on SLOT2 killing SLOT3 other people and injuring wounding SLOT4 end detroit the *e* a s *e* building buildingin detroit flattened ground levelled to blasted leveled *e* was reduced razed leveled to down rubble into ashes *e* to *e* (1) (2) Figure 1: Examples of paraphrase patterns extracted by Barzilay and Lee (2003) and Pang et al.
---------------------------------------------------
W09-2805:11	52:188	A few unsupervised metrics have been applied to automatic paraphrase identification and extraction (Barzilay & Lee, 2003; Dolan et al., 2004).
---------------------------------------------------
W09-2805:12	53:188	However, these unsupervised methodologies show a major drawback by extracting quasi-exact or even exact match pairs of sentences as they rely on classical string similarity measures such as the Edit Distance in the case of (Dolan et al., 2004) and Word N-gram Overlap for (Barzilay & Lee, 2003).
---------------------------------------------------
C08-1107:13	37:203	Some works focused on learning rules from comparable corpora, containing comparable documents such as different news articles from the same date on the same topic (Barzilay and Lee, 2003; Ibrahim et al., 2003).
---------------------------------------------------
C08-1029:14	35:200	Multiple translations of the same text (Barzilay and McKeown, 2001), corresponding articles from multiple news sources (Barzilay and Lee, 2003; Quirk et al., 2004; Dolan et al., 2004), and bilingual corpus (Bannard and Callison-Burch, 2005) have been utilized.
---------------------------------------------------
W07-1425:15	12:188	The third estimates the equivalence based on word alignment composed using templates or translation probabilities derived from a set of parallel text (Barzilay and Lee, 2003; Brockett and Dolan, 2005).
---------------------------------------------------
P06-2070:16	77:155	If we consider these probabilities as a vector, the similarities of two English words can be obtained by computing the dot product of their corresponding vectors.2 The formula is described below: similarity(ei, ej) = Nsummationdisplay k=1 p(ei|fk)p(ej|fk) (3) Paraphrasing methods based on monolingual parallel corpora such as (Pang et al. , 2003; Barzilay and Lee, 2003) can also be used to compute the similarity ratio of two words, but they dont have as rich training resources as the bilingual methods do.
---------------------------------------------------
W03-1608:17	131:181	Similar to the work of Barzilay and Lee (2003), who have applied paraphrase generation techniques to comparable corpora consisting of different newspaper articles about the same event, we are currently attempting to solve the data sparseness problem by extending our approach to non-parallel corpora.
---------------------------------------------------
E09-1082:18	80:216	(2006) propose using a statistical word alignment algorithm as a more robust way of aligning (monolingual) outputs into a confusion network for system com2Barzilay and Lee (2003) construct lattices over paraphrases using an iterative pairwise multiple sequence alignment (MSA) algorithm.
---------------------------------------------------
W07-1424:19	12:170	Most of the reported work on paraphrase generation from arbitrary input sentences uses machine learning techniques trained on sentences that are known or can be inferred to be paraphrases of each other (Bannard and Callison-Burch, 2005; Barzilay and Lee, 2003; Barzilay and McKeown, 2001; Callison-Burch et al. , 2006; Dolan et al. , 2004; Ibrahim et al. , 2003; Lin and Pantel, 2001; Pang et al. , 2003; Quirk et al. , 2004; Shinyama et al. , 2002).
---------------------------------------------------
P07-1058:20	56:205	2.2 Evaluation of Acquisition Algorithms Many methods for automatic acquisition of rules have been suggested in recent years, ranging from distributional similarity to finding shared contexts (Lin and Pantel, 2001; Ravichandran and Hovy, 2002; Shinyama et al. , 2002; Barzilay and Lee, 2003; Szpektor et al. , 2004; Sekine, 2005).
---------------------------------------------------
P07-1058:21	75:205	Indeed, only few earlier works reported inter-judge agreement level, and those that did reported rather low Kappa values, such as 0.54 (Barzilay and Lee, 2003) and 0.55 0.63 (Szpektor et al. , 2004).
---------------------------------------------------
P07-1058:22	66:205	Indeed, the prominent approach for evaluating the quality of rule acquisition algorithms is by human judgment of the learned rules (Lin and Pantel, 2001; Shinyama et al. , 2002; Barzilay and Lee, 2003; Pang et al. , 2003; Szpektor et al. , 2004; Sekine, 2005).
---------------------------------------------------
C04-1051:23	8:175	1 Introduction The importance of learning to manipulate monolingual paraphrase relationships for applications like summarization, search, and dialog has been highlighted by a number of recent efforts (Barzilay & McKeown 2001; Shinyama et al. 2002; Lee & Barzilay 2003; Lin & Pantel 2001).
---------------------------------------------------
C04-1051:24	26:175	While the idea of exploiting multiple news reports for paraphrase acquisition is not new, previous efforts (for example, Shinyama et al. 2002; Barzilay and Lee 2003) have been restricted to at most two news sources.
---------------------------------------------------
C04-1051:25	14:175	Lee & Barzilay (2003), for example, use MultiSequence Alignment (MSA) to build a corpus of paraphrases involving terrorist acts.
---------------------------------------------------
C04-1051:26	155:175	Mean number of instances of paraphrase phenomena per sentence (such as Multiple Sequence Alignment, as employed by Barzilay & Lee 2003).
---------------------------------------------------
P04-2006:27	114:146	Barzilay & Lee (2003) also identify paraphrases in their paraphrased sentence generation system.
---------------------------------------------------
W09-0604:28	131:186	3.4 Perspectives for automatic paraphrase extraction There is a growing amount of work on automatic extraction of paraphrases from text corpora (Lin and Pantel, 2001; Barzilay and Lee, 2003; Ibrahim et al., 2003; Dolan et al., 2004).
---------------------------------------------------
W06-1603:29	50:249	Barzilay and Lee (2003) proposed to apply multiple-sequence alignment (MSA) for traditional, sentence-level PR.
---------------------------------------------------
W07-1429:30	13:169	Second, we will discuss the work done by (Barzilay & Lee, 2003) who use clustering of paraphrases to induce rewriting rules.
---------------------------------------------------
W07-1429:31	4:169	Second, we discuss the work done by (Barzilay & Lee, 2003) who use clustering of paraphrases to induce rewriting rules.
---------------------------------------------------
W07-1429:32	43:169	However, these unsupervised methodologies show a major drawback by extracting quasi-exact2 or even exact match pairs of sentences as they rely on classical string similarity measures such as the Edit Distance in the case of (Dolan & Brockett, 2004) and word N-gram overlap for (Barzilay & Lee, 2003).
---------------------------------------------------
W07-1429:33	89:169	These results confirm the observed figures in the previous subsection and reinforce the sight that clustering is a worthless effort for automatic paraphrase corpora construction, contrarily to what (Barzilay & Lee, 2003) suggest.
---------------------------------------------------
W07-1429:34	30:169	Comparatively, (Barzilay & Lee, 2003) propose to use the N-gram Overlap metric to capture similarities between sentences and automatically create paraphrase corpora.
---------------------------------------------------
W07-1429:35	8:169	Sentence Compression takes an important place for Natural Language Processing (NLP) tasks where specific constraints must be satisfied, such as length in summarization (Barzilay & Lee, 2002; Knight & Marcu, 2002; Shinyama et al. , 2002; Barzilay & Lee, 2003; Le Nguyen & Ho, 2004; Unno et al. , 2006), style in text simplification (Marsi & Krahmer, 2005) or sentence simplification for subtitling (Daelemans et al. , 2004).
---------------------------------------------------
W07-1429:36	70:169	Table 2: Figures about clustering algorithms Algorithm # Sentences/# Clusters S-HAC 6,23 C-HAC 2,17 QT 2,32 EM 4,16 In fact, table 2 shows that most of the clusters have less than 6 sentences which leads to question the results presented by (Barzilay & Lee, 2003) who only keep the clusters that contain more than 10 sentences.
---------------------------------------------------
W07-1429:37	17:169	As our work is based on the first paradigm, we will focus on the works proposed by (Barzilay & Lee, 2003) and (Le Nguyen & Ho, 2004).
---------------------------------------------------
W07-1429:38	42:169	3.1 Paraphrase Identification A few unsupervised metrics have been applied to automatic paraphrase identification and extraction (Barzilay & Lee, 2003; Dolan & Brockett, 2004).
---------------------------------------------------
W07-1429:39	16:169	2 Related Work Two different approaches have been proposed for Sentence Compression: purely statistical methodologies (Barzilay & Lee, 2003; Le Nguyen & Ho, 2004) and hybrid linguistic/statistic methodologies (Knight & Marcu, 2002; Shinyama et al. , 2002; Daelemans et al. , 2004; Marsi & Krahmer, 2005; Unno et al. , 2006).
---------------------------------------------------
W07-1429:40	55:169	However, as (Barzilay & Lee, 2003) do not propose any evaluation of which clustering algorithm should be used, we experiment a set of clustering algorithms and present the comparative results.
---------------------------------------------------
W07-1429:41	18:169	(Barzilay & Lee, 2003) present a knowledge-lean algorithm that uses multiple-sequence alignment to 177 learn generate sentence-level paraphrases essentially from unannotated corpus data alone.
---------------------------------------------------
W07-1429:42	165:169	Experiments, by using 4 algorithms and through visualization techniques, revealed that clustering is a worthless effort for paraphrase corpora construction, contrary to the literature claims (Barzilay & Lee, 2003).
---------------------------------------------------
W07-1429:43	53:169	On one hand, as (Barzilay & Lee, 2003) evidence, clusters of paraphrases can lead to better learning of text-totext rewriting rules compared to just pairs of paraphrases.
---------------------------------------------------
W07-1429:44	33:169	Unlike (Le Nguyen & Ho, 2004), one interesting idea proposed by (Barzilay & Lee, 2003) is to cluster similar pairs of paraphrases to apply multiplesequence alignment.
---------------------------------------------------
W07-1429:45	51:169	In particular, it shows systematically better F-Measure and Accuracy measures over all other metrics showing an improvement of (1) at least 2.86% in terms of F-Measure and 3.96% in terms of Accuracy and (2) at most 6.61% in terms of FMeasure and 6.74% in terms of Accuracy compared to the second best metric which is also systematically the word N-gram overlap similarity measure used by (Barzilay & Lee, 2003).
---------------------------------------------------
I08-1070:46	47:198	The other utilizes a sort of parallel texts, such as multiple translation of the same text (Barzilay and McKeown, 2001; Pang et al., 2003), corresponding articles from multiple news sources (Barzilay and Lee, 2003; Dolan et al., 2004), and bilingual corpus (Wu and Zhou, 2003; Bannard and Callison-Burch, 2005).
---------------------------------------------------
I05-5004:47	67:184	Some studies exploit topically related articles derived from multiple news sources (Barzilay and Lee, 2003; Shinyama and Sekine, 2003; Quirk et al. , 2004; Dolan et al. , 2004).
---------------------------------------------------
N04-1015:48	61:174	But because we want the insertion state a1a16a20 to model digressions or unseen topics, we take the novel step of forcing its language model to be complementary to those of the other states by setting a2 a3a27a38 a21 a8 a8 a4 a8 a24 a26a11a28a30a29a6 a39a41a40a43a42a45a44a16a46 a1a48a47a1a50a49 a20 a2 a3 a26a17a21 a8a9a8 a4 a8 a24 a51a53a52a55a54a57a56 a21 a39a58a40a43a42a45a44a16a46 a1a59a47a1a50a49 a20 a2 a3a27a26a11a21a50a60 a4 a8 a24a30a24 a17 4Following Barzilay and Lee (2003), proper names, numbers and dates are (temporarily) replaced with generic tokens to help ensure that clusters contain sentences describing the same event type, rather than same actual event.
---------------------------------------------------
P09-3004:49	147:164	In another generation approach, Barzilay and Lee (2002; 2003) look for pairs of slotted word lattices that share many common slot fillers; the lattices are generated by applying a multiplesequence alignment algorithm to a corpus of multiple news articles about the same events.
---------------------------------------------------
P08-1077:50	34:235	(2004) and Barzilay and Lee (2003) used comparable news articles to obtain sentence level paraphrases.
---------------------------------------------------
W06-1403:51	8:180	Our experience suggests that disjunctive LFs are an important capability, especially as one seeks to make grammars reusable across applications, and to employ domain-specific, sentence-level paraphrases (Barzilay and Lee, 2003).
---------------------------------------------------
N06-1058:52	54:238	2 Related Work Automatic Paraphrasing and Entailment Our work is closely related to research in automatic paraphrasing, in particular, to sentence level paraphrasing (Barzilay and Lee, 2003; Pang et al. , 2003; Quirk et al. , 2004).
---------------------------------------------------
N06-1058:53	101:238	2This can explain why previous attempts to use WordNet for generating sentence-level paraphrases (Barzilay and Lee, 2003; Quirk et al. , 2004) were unsuccessful.
---------------------------------------------------
W04-0910:54	17:298	Similarly, (Barzilay and Lee, 2003) and (Shinyanma et al. , 2002) learn sentence level paraphrase templates from a corpus of news articles stemming from different news source.
---------------------------------------------------
P06-2096:55	38:213	2 Related work Our work is closest in spirit to the two papers that inspired us (Barzilay and Lee, 2003) and (Pang et al. , 2003).
---------------------------------------------------
P06-2096:56	12:213	Previous work aligns a group of sentences into a compact word lattice (Barzilay and Lee, 2003), a finite state automaton representation that can be used to identify commonality or variability among comparable texts and generate paraphrases.
---------------------------------------------------
I08-2110:57	18:139	Recently, some work has been done on corpusbased paraphrase extraction (Lin and Pantel, 2001; Barzilay and Lee, 2003).
---------------------------------------------------
I05-5001:58	6:155	A growing body of recent research has focused on the problems of identifying and generating paraphrases, e.g., Barzilay & McKeown (2001), Lin & Pantel (2002), Shinyama et al, (2002), Barzilay & Lee (2003), and Pang et al.
---------------------------------------------------
I05-5001:59	23:155	Barzilay & Lee (2003) employ Multiple Sequence Alignment (MSA, e.g., Durbin et al. , 1998) to align strings extracted from closely related news articles.
---------------------------------------------------
I05-5001:60	27:155	The word-based edit distance heuristic yields pairs that are relatively clean but offer relatively minor rewrites in generation, especially when compared to the MSA model of (Barzilay & Lee, 2003).
---------------------------------------------------
I05-5001:61	85:155	Barzilay & Lee (2003) and Quirk et al.
---------------------------------------------------
P05-1074:62	30:147	2 Extracting paraphrases Much previous work on extracting paraphrases (Barzilay and McKeown, 2001; Barzilay and Lee, 2003; Pang et al. , 2003) has focused on finding identifying contexts within aligned monolingual sentences from which divergent text can be extracted, and treated as paraphrases.
---------------------------------------------------
P05-1074:63	14:147	Past work (Barzilay and McKeown, 2001; Barzilay and Lee, 2003; Pang et al. , 2003; Ibrahim et al. , 2003) has examined the use of monolingual parallel corpora for paraphrase extraction.
---------------------------------------------------
W05-1210:64	80:142	Such transformations are typically denoted as paraphrases in the literature, where a wealth of methods for their automatic acquisition were proposed (Lin and Pantel, 2001; Shinyama et al. , 2002; Barzilay and Lee, 2003; Szpektor et al. , 2004).
---------------------------------------------------
W08-0906:65	43:177	There exist many different string similarity measures: word overlap (Tomuro and Lytinen, 2004), longest common subsequence (Islamand Inkpen,2007), Levenshteinedit distance (Dolan et al., 2004), word n-gramoverlap (Barzilay and Lee, 2003) etc. Semantic similarity measures are obtained by first computing the semantic similarity of the words containedin the sentencesbeing compared.
---------------------------------------------------
W08-0906:66	40:177	While word and phrasal paraphrases can be assimilated to the well-studied notion of synonymy, sentencelevel paraphrasingis moredifficult to grasp and cannot be equated with word-for-word or phrase-by-phrase substitution since it might entail changes in the structure of the sentence (Barzilay and Lee, 2003).
---------------------------------------------------
W08-0906:67	115:177	In order to be able to compare the edit distance with the other metrics, we have used the following formula(Wen et al., 2002)whichnormalisesthe minimum edit distance by the length of the longest questionand transformsit into a similaritymetric: normalisededitdistance = 1 edit dist(q1,q2)max(| q 1 |,| q2 |) Word Ngram Overlap This metric compares the word n-gramsin both questions: ngramoverlap = 1N Nsummationdisplay n=1 | Gn(q1)  Gn(q2) | min(| Gn(q1) |,| Gn(q2) |) where Gn(q) is the set of n-grams of length n in question q and N usually equals 4 (Barzilay and Lee, 2003;Cordeiroet al., 2007).
---------------------------------------------------
N04-1031:68	54:210	There are several works that try to learn paraphrase pairs from parallel or comparable corpora (Barzilay and McKeown, 2001; Shinyama et al. , 2002; Barzilay and Lee, 2003; Pang et al. , 2003).
---------------------------------------------------
N04-1031:69	44:210	Although a large number of studies have been made on learning paraphrases, for example (Barzilay and Lee, 2003), there are only a few studies which address the connotational difference of paraphrases.
---------------------------------------------------
I05-5002:70	9:264	g2 2 Motivation The success of Statistical Machine Translation (SMT) has sparked a successful line of investigation that treats paraphrase acquisition and generation essentially as a monolingual machine translation problem (e.g. , Barzilay & Lee, 2003; Pang et al. , 2003; Quirk et al. , 2004; Finch et al. , 2004).
---------------------------------------------------
P08-1116:71	58:268	Different news articles reporting on the same event are commonly used as monolingual comparable corpora, from which both paraphrase patterns and phrasal paraphrases can be derived (Shinyama et al., 2002; Barzilay and Lee, 2003; Quirk et al., 2004).
---------------------------------------------------
P08-1116:72	73:268	For example, Barzilay and Lee (2003) applied multiple-sequence alignment (MSA) to parallel news sentences and induced paraphrasing patterns for generating new sentences.
---------------------------------------------------
P08-1116:73	117:268	3 Monolingual comparable corpus: Similar to the methods in (Shinyama et al., 2002; Barzilay and Lee, 2003), we construct a corpus of comparable documents from a large corpus D of news articles.
---------------------------------------------------
N06-1008:74	20:225	Previous attempts have used, for instance, the similarities between case frames (Lin and Pan57 tel, 2001), anchor words (Barzilay and Lee, 2003; Shinyama et al. , 2002; Szepektor et al. , 2004), and a web-based method(Szepektor et al. , 2004;Geffet and Dagan, 2005).
---------------------------------------------------
N03-1024:75	51:235	Its still possible to use MSA if, for example, the input is pre-clustered to have the same constituent ordering (Barzilay and Lee (2003)).
---------------------------------------------------
P04-1077:76	206:211	Paraphrases can also be automatically acquired using statistical methods as shown by Barzilay and Lee (2003).
---------------------------------------------------
D09-1122:77	30:204	2 Related Work Previous studies on entailment, inference rules, and paraphrase acquisition are roughly classified into those that require comparable corpora (Shinyama et al., 2002; Barzilay and Lee, 2003; Ibrahim et al., 2003) and those that do not (Lin and Pantel, 2001; Weeds and Weir, 2003; Geffet and Dagan, 2005; Pekar, 2006; Bhagat et al., 2007; Szpektor and Dagan, 2008).
---------------------------------------------------
D09-1122:78	34:204	Barzilay and Lee (2003) also used newspaper articles on the same event as comparable corpora to acquire paraphrases.
---------------------------------------------------
W03-1602:79	26:182	(Barzilay and McKeown, 2001; Shinyama et al. , 2002; Barzilay and Lee, 2003).
---------------------------------------------------
P09-1053:80	9:248	For natural language engineers, the problem bears on information management systems like abstractive summarizers that must measure semantic overlap between sentences (Barzilay and Lee, 2003), question answering modules (Marsi and Krahmer, 2005) and machine translation (Callison-Burch et al., 2006).
---------------------------------------------------
W03-1605:81	7:187	For this reason, paraphrase poses a great challenge for many Natural Language Processing (NLP) tasks, just as ambiguity does, notably in text summarization and NL generation (Barzilay and Lee, 2003; Pang et al. , 2003).
---------------------------------------------------
W08-1911:82	133:160	Barzilay and Lee (Barzilay and Lee, 2003) learned paraphrasing patterns as pairs of word lattices, which are then used to produce sentence level paraphrases.
---------------------------------------------------
W07-0909:83	76:167	Automatically Learning Entailment Rules from the Web Many algorithms for automatically learning paraphrases and entailment rules have been explored in recent years (Lin and Pantel, 2001; 1http://jakarta.apache.org/lucene/docs/index.html 67 Ravichandran and Hovy, 2002; Shinyama et al. , 2002; Barzilay and Lee, 2003; Sudo et al. , 2003; Szpektor et al. , 2004; Satoshi, 2005).
---------------------------------------------------
C08-1110:84	29:196	This is related to the wellstudied problem of identifying paraphrases (Barzilay and Lee, 2003; Pang et al., 2003) and the more general variant of recognizing textual entailment, which explores whether information expressed in a hypothesis can be inferred from a given premise.
---------------------------------------------------
I05-5007:85	13:203	Generation of paraphrase examples was also investigated (Barzilay and Lee, 2003; Quirk et al. , 2004).
---------------------------------------------------
N09-3008:86	18:144	The use of Profile HMMs for multiple sequence alignment also presents applications to the acquisition of mapping dictionaries (Barzilay and Lee, 2002) and sentence-level paraphrasing (Barzilay and Lee, 2003).
---------------------------------------------------
P09-2063:87	9:121	For instance, automatic summary can be seen as a particular paraphrasing task (Barzilay and Lee, 2003) with the aim of selecting the shortest paraphrase.
---------------------------------------------------
P06-1034:88	180:231	5 Related Work Automatically finding sentences with the same meaning has been extensively studied in the field of automatic paraphrasing using parallel corpora and corporawith multiple descriptionsof the same events (Barzilay and McKeown, 2001; Barzilay and Lee, 2003).
---------------------------------------------------
P09-1094:89	22:246	Some researchers then tried to automatically extract paraphrase rules (Lin and Pantel, 2001; Barzilay and Lee, 2003; Zhao et al., 2008b), which facilitates the rule-based PG methods.
---------------------------------------------------
I05-5008:90	86:211	Such a method alleviates the problem of creating templates from examples which would be used in an ulterior phase of generation (BARZILAY and LEE, 2003).
---------------------------------------------------
