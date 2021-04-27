P09-1024:1	95:261	There has been a sizable amount of research on structure induction ranging fromlinearsegmentation(Hearst, 1994)tocontent modeling (Barzilay and Lee, 2004).
---------------------------------------------------
N06-2003:2	22:93	Content-oriented models, such as (Barzilay and Lee, 2004), rely on the re-occurrence of patterns of topics over multiple realizations of thematically similar discourses, such as a series of newspaper articles about similar events.
---------------------------------------------------
N09-1042:3	29:215	For instance, content models (Barzilay and Lee, 2004; Elsner et al., 2007) are implemented as HMMs, where the states correspond to topics of domain-specific information, and transitions reflect pairwise ordering preferences.
---------------------------------------------------
N07-1055:4	159:190	The HMM content model of (Barzilay and Lee, 2004), which does have global structure, performs much better on ordering, at  of .44.
---------------------------------------------------
N07-1055:5	123:190	Unlike Barzilay and Lee (2004), we do not initialize with an informative starting distribution.
---------------------------------------------------
N07-1055:6	143:190	(%) (Barzilay and Lapata, 2005) 90 (Barzilay and Lee, 2004) .44 745 (Soricut and Marcu, 2006) .50 -6 Topic-based (relaxed) .50 94 Table 1: Results for AIRPLANE test data.
---------------------------------------------------
N07-1055:7	103:190	Like Barzilay and Lee (2004), we learn an HMM in which each sentence has a hidden topic qi, which is chosen conditioned on the previous state qi1.
---------------------------------------------------
N07-1055:8	2:190	c2007 Association for Computational Linguistics A Unified Local and Global Model for Discourse Coherence Micha Elsner, Joseph Austerweil, and Eugene Charniak Brown Laboratory for Linguistic Information Processing (BLLIP) Brown University Providence, RI 02912 {melsner,ec}@cs.brown.edu, joseph.austerweil@gmail.com Abstract We present a model for discourse coherence which combines the local entitybased approach of (Barzilay and Lapata, 2005) and the HMM-based content model of (Barzilay and Lee, 2004).
---------------------------------------------------
N07-1055:9	20:190	Our global model is an HMM like that of Barzilay and Lee (2004), but with emission probabilities drawn from the entity grid.
---------------------------------------------------
N07-1055:10	164:190	(Barzilay and Lee, 2004), which lacks local features, does quite poorly on this task (74%), while our model performs extremely well (94%).
---------------------------------------------------
N07-1055:11	175:190	Their mixture includes an entity grid model and a version of the HMM of (Barzilay and Lee, 2004), which uses n-gram language modeling.
---------------------------------------------------
N07-1055:12	137:190	5.1 Sentence Ordering In the sentence ordering task, (Lapata, 2003; Barzilay and Lee, 2004; Barzilay and Lapata, 2005; Soricut and Marcu, 2006), we view a document as an unordered bag of sentences and try to find the ordering of the sentences which maximizes coherence according to our model.
---------------------------------------------------
N07-1055:13	13:190	In contrast, the global HMM model of Barzilay and Lee (2004) tries to track the predictable changes in topic between sentences.
---------------------------------------------------
N07-1055:14	130:190	5 Experiments Our experiments use the popular AIRPLANE corpus, a collection of documents describing airplane crashes taken from the database of the National 440 Transportation Safety Board, used in (Barzilay and Lee, 2004; Barzilay and Lapata, 2005; Soricut and Marcu, 2006).
---------------------------------------------------
N07-1055:15	151:190	This task bears some resemblance to the task of discriminating coherent from incoherent essays in (Miltsakaki and Kukich, 2004), and is also equivalent in the limit to the ranking metric of (Barzilay and Lee, 2004), which we cannot calculate because our model does not produce k-best output.
---------------------------------------------------
P05-1018:16	109:224	4.1 Text Ordering Text structuring algorithms (Lapata, 2003; Barzilay and Lee, 2004; Karamanis et al. , 2004) are commonly evaluated by their performance at information-ordering.
---------------------------------------------------
D07-1009:17	183:246	Relative to other corpora used in text structuring research (Barzilay and Lee, 2004; Lapata, 2003; Karamanis et al. , 2004), texts in 6Insertion is only one type of recorded update, others include deletions and sentence rewriting.
---------------------------------------------------
P05-1046:18	184:198	More recently, Barzilay and Lee (2004) defined content models, which can be viewed as field segmentation models occurring at the level of discourse.
---------------------------------------------------
W09-2103:19	24:138	Such probablistic inference of discourse structure has been used in recent work with HMMs for topic identification (Barzilay & Lee 2004) and related graphical models for segmenting multi-party spoken discourse (Purver et al. 2006).
---------------------------------------------------
P08-2011:20	78:98	Previous work has focused on the AIRPLANE corpus (Barzilay and Lee, 2004), which contains short announcements of airplane crashes written by and for domain experts.
---------------------------------------------------
P06-1003:21	22:134	Recent advances have used generative models, allowing lexical models of the topics themselves to be built while segmenting (Imai et al. , 1997; Barzilay and Lee, 2004), and we take a similar approach here, although with some important differences detailed below.
---------------------------------------------------
P06-1003:22	131:134	We are also investigating improvements into the lexical model as presented here, firstly via simple techniques such as word stemming and replacement of named entities by generic class tokens (Barzilay and Lee, 2004); but also via the use of multiple ASR hypotheses by incorporating word confusion networks into our model.
---------------------------------------------------
P06-1003:23	95:134	representation, however, might improve its performance (Barzilay and Lee, 2004), although similar benefits might equally apply to our model.
---------------------------------------------------
P06-1003:24	59:134	Our algorithm is related to (Barzilay and Lee, 2004)s approach to text segmentation, which uses ahiddenMarkovmodel(HMM)tomodelsegmentation and topic inference for text using a bigram representation in restricted domains.
---------------------------------------------------
P08-1092:25	183:209	Barzilay and Lee (2004) use HMMs to capture topic shift within a particular domain; sequence of topic shifts then guides the subsequent ordering of sentences within the summary.
---------------------------------------------------
P08-1092:26	86:209	Similar to (Barzilay and Lee, 2004), we automatically learn how to order our biographical sentences by observing the typical order of presentation of information in a particular domain.
---------------------------------------------------
D08-1057:27	104:247	Like Barzilay and Lee (2004), this model was used to order extracted sentences in summaries.
---------------------------------------------------
D08-1057:28	99:247	Barzilay and Lee (2004) showed that it is possible to obtain schema-like knowledge automatically from a corpus for the purposes of extracting sentences and ordering them.
---------------------------------------------------
D08-1057:29	63:247	Corpus-based methods inspired by the notion of schemata have been explored in the past by Lapata (2003) and Barzilay and Lee (2004) for ordering sentences extracted in a multi-document summarisation application.
---------------------------------------------------
N09-1041:30	128:207	Reflecting intuition that adjacent sentences are likely to share specific content vocabulary, we utilize a sticky HMM as in Barzilay and Lee (2004) over the each sentences ZS.
---------------------------------------------------
N09-1041:31	11:207	Another strand of work (Barzilay and Lee, 2004; Daume III and Marcu, 2006; Eisenstein and Barzilay, 2008), has explored the use of structured probabilistic topic models to represent document content.
---------------------------------------------------
N09-1041:32	108:207	However, as Barzilay and Lee (2004) observe, the content of document collections is highly structured, consistingofseveraltopicalthemes, each with its own vocabulary and ordering preferences.
---------------------------------------------------
J06-4002:33	264:281	Barzilay and Lee (2004) and Barzilay and Lapata (2005) measure accuracy as the percentage of test items for which the system gives preference to the gold-standard reference order.
---------------------------------------------------
J06-4002:34	27:281	The task concerns finding an acceptable ordering for a set of preselected information-bearing items (Lapata 2003; Barzilay and Lee 2004).
---------------------------------------------------
J06-4002:35	266:281	Barzilay and Lee (2004) propose an additional evaluation measure based on ranks.
---------------------------------------------------
J06-4002:36	108:281	The texts were randomly sampled from a corpus collected by Barzilay and Lee (2004) (sampling took place over eight-sentence-length documents only).
---------------------------------------------------
J06-4002:37	29:281	Depending on the application and domain at hand, the items to be ordered may vary greatly from propositions (Karamanis 2003; Dimitromanolaki and Androutsopoulos 2003) to trees (Mellish et al. 1998) or sentences (Lapata 2003; Barzilay and Lee 2004).
---------------------------------------------------
J06-4002:38	253:281	Furthermore, statistical generation systems (Lapata 2003; Barzilay and Lee 2004; Karamanis and Manurung 2002; Mellish et al. 1998) could use  as a means of directly optimizing information ordering, much in the same way MT systems optimize model parameters using BLEU as a measure of translation quality (Och 2003).
---------------------------------------------------
J06-4002:39	50:281	They can be facts in a database (Duboue and McKeown 2001), propositions (Karamanis 2003), discourse trees (Mellish et al. 1998), or sentences (Lapata 2003; Barzilay and Lee 2004).
---------------------------------------------------
J06-4002:40	33:281	Indeed, several studies have adopted Kendalls  as a performance measure for evaluating the output of information-ordering components both in the context of concept-to-text generation (Karamanis and Mellish 2005; Karamanis 2003) and summarization (Lapata 2003; Barzilay and Lee 2004; Okazaki, Matsuo, and Ishizuka 2004).
---------------------------------------------------
D08-1100:41	167:258	For HMM-based segmentation, we modified Barzilay and Lees (2004) content models by using larger text spans when inducing the HMM states.
---------------------------------------------------
D08-1100:42	151:258	Two unsupervised discourse segmentation algorithms are investigated: TextTiling (Hearst, 1997) and Hidden Markov Modeling (Barzilay and Lee, 2004).
---------------------------------------------------
P06-2097:43	15:192	In the field of Natural Language Processing, Barzilay and Lee have recently proposed a probabilistic content model for representing topics and topic shifts (Barzilay and Lee, 2004).
---------------------------------------------------
P06-2097:44	43:192	In contrast, Barzilay and Lee modeled a content structure of texts within specific domains, such as earthquake and finance (Barzilay and Lee, 2004).
---------------------------------------------------
J09-1003:45	200:302	38 Karamanis et al. Centering for Information Ordering often the most viable alternative (Poesio et al. 2004; Barzilay and Lee 2004).
---------------------------------------------------
J09-1003:46	244:302	Barzilay and Lee (2004) introduce a stochastic model for information ordering which computes the probability of generating the OSO and every alternative ordering.
---------------------------------------------------
J09-1003:47	203:302	This methodology is based on the premise that the original sentence order (OSO, Barzilay and Lee 2004) observed in a corpus text is more coherent than any other ordering.
---------------------------------------------------
J09-1003:48	4:302	1.Introduction Information ordering (Barzilay and Lee 2004), that is, deciding in which sequence to present a set of preselected information-bearing items, has received much attention in recent work in automatic text generation.
---------------------------------------------------
J09-1003:49	252:302	On the other hand, Barzilay and Lee (2004) enumerate exhaustively each possible ordering, which might become impractical as the search space grows factorially.
---------------------------------------------------
J09-1003:50	257:302	16 Neither Barzilay and Lapata (2005) nor Barzilay and Lee (2004) appear to consider the possibility that two orderings may be equally ranked.
---------------------------------------------------
J09-1003:51	35:302	1 2.InformationOrdering Information ordering has been investigated by substantial recent work in text-totext generation (Barzilay, Elhadad, and McKeown 2002; Lapata 2003; Barzilay and Lee 2004; Barzilay and Lapata 2005; Bollegala, Okazaki, and Ishizuka 2006; Ji and Pulman 2006; Siddharthan 2006; Soricut and Marcu 2006; Madnani et al. 2007, among others) as well as concept-to-text generation (particularly Kan and McKeown [2002] and Dimitromanolaki and Androutsopoulos 2003).
---------------------------------------------------
P06-2103:52	140:271	On the other hand, we achieve the highest accuracy figure (0.50) on the ACCIDENTS corpus, outperforming the previous-highest figure (0.44) of Barzilay and Lee (2004).
---------------------------------------------------
P06-2103:53	147:271	At the other end of the spectrum, the exhaustive search of Barzilay and Lee (2004), while ensuring optimal solutions, is prohibitively expensive, and cannot be used to perform utility-based training.
---------------------------------------------------
P06-2103:54	97:271	4.1 Evaluation setting The task on which we conduct our evaluation is information ordering (Lapata, 2003; Barzilay and Lee, 2004; Barzilay and Lapata, 2005).
---------------------------------------------------
P06-2103:55	111:271	4.2 Evaluation of Search Algorithms We evaluated the performance of several search algorithms across four stochastic models of document coherence: the IBMa0 a3 and IBM a15 a3 coherence models, the content model of Barzilay and Lee (2004) (CM), and the entity-based model of Barzilay and Lapata (2005) (EB) (Section 2).
---------------------------------------------------
P06-2103:56	136:271	We first note that, unfortunately, we failed to accurately reproduce the model of Barzilay and Lee (2004).
---------------------------------------------------
P06-2103:57	9:271	In contrast, more recent research has focused on stochastic approaches that model discourse coherence at the local lexical (Lapata, 2003) and global levels (Barzilay and Lee, 2004), while preserving regularities recognized by classic discourse theories (Barzilay and Lapata, 2005).
---------------------------------------------------
P06-2103:58	113:271	Lapata (2003) 0.48 0.07 Barzilay & Lee (2004) 0.81 0.44 Barzilay & Lee (reproduced) 0.39 0.36 Barzilay & Lapata (2005) 0.19 0.12 IBMa0 a3, IDL-CH-HB a23 a25a26a25 0.38 0.41 Log-lina10a13a12a15a14a22a16a9a18a20a19, IDL-CH-HB a23 a25a26a25 0.47 0.50 Table 3: Comparison of overall performance (affected by both model & search procedure) of our framework with previous results.
---------------------------------------------------
P06-2103:59	139:271	The large difference on the EARTHQUAKEScorpus between the performance of Barzilay and Lee (2004) and our reproduction of their model is responsible for the overall lower performance (0.47) of our log-lineara10a13a12a15a14a17a16a9a18a20a19 model and IDL-CH-HBa3 a56a72a56 search algorithm, which is nevertheless higher than that of its component model CM (0.39).
---------------------------------------------------
P06-2103:60	13:271	A frequently used testbed for coherence models is the discourse ordering problem, which occurs often in text generation, complex question answering, and multi-document summarization: given a0 discourse units, what is the most coherent ordering of them (Marcu, 1996; Lapata, 2003; Barzilay and Lee, 2004; Barzilay and Lapata, 2005)?
---------------------------------------------------
P06-2103:61	41:271	2.2 Global Models of Discourse Coherence Barzilay and Lee (2004) propose a document content model that uses a Hidden Markov Model 804 (HMM) to capture more global aspects of coherence.
---------------------------------------------------
P06-2103:62	65:271	They unfold an input IDLgraph on-the-fly, as follows: starting from the initial vertex a25 a40, the input graph is traversed in an IDL-specific manner, by creating states which 2Following Barzilay and Lee (2004), proper names, dates, and numbers are replaced with generic tokens.
---------------------------------------------------
P06-2103:63	83:271	One of the most frequently used metrics for the automatic evaluation of document coherence is Kendalls a0 (Lapata, 2003; Barzilay and Lee, 2004).
---------------------------------------------------
P06-2103:64	133:271	We are able to provide this comparison based on the TAU figures reported in (Barzilay and Lee, 2004).
---------------------------------------------------
W09-2807:65	104:117	The insertion in the abstract of linguistic material not present in the input document has been addressed in paraphrase generation (Barzilay and Lee, 2004) and canned-based summarization (Oakes and Paice, 2001) in limited domains.
---------------------------------------------------
N09-3014:66	155:162	When compared to the results obtained by Barzilay and Lapata (2008) and Barzilay and Lee (2004), it would appear that direct sentenceto-sentence similarity (as suggested by the Barzilay and Lapata baseline score) or capturing topic sequences are essential for acquiring the correct sequence of sentences in the earthquake dataset.
---------------------------------------------------
N09-3014:67	135:162	Coreference+Syntax+Salience+ and Coreference?Syntax+Salience+ are the Barzilay and Lapata (2008) model, HMM-based Content Models is the Barzilay and Lee (2004) paper and Latent Semantic Analysis is the Barzilay and Lapata (2008) implementation of Peter W. Foltz and Landauer (1998).
---------------------------------------------------
N09-3014:68	32:162	This is specific to their approach as both Lapata (2003)s and Barzilay and Lee (2004)s approaches are not tailored to summarization and therefore do not experience the topic bias problem.
---------------------------------------------------
N09-3014:69	12:162	The most prominent approaches include: (1) temporal ordering in terms of publication date (Barzilay, 2003), (2) temporal ordering in terms of textual cues in sentences (Bollegala et al., 2006), (3) the topic of the sentences (Barzilay, 2003), (4) coherence theories (Barzilay and Lapata, 2008), e.g., Centering Theory, (5) content models (Barzilay and Lee, 2004), and (6) ordering(s) in the underlying documents in the case of summarisation (Bollegala et al., 2006; Barzilay, 2003).
---------------------------------------------------
N09-3014:70	23:162	Barzilay and Lee (2004)s knowledge-lean approach attempts to automate the inference of knowledge-rich information using a distributional view of content.
---------------------------------------------------
N06-2017:71	15:80	The same data and similar methods were used by Barzilay and Lee (2004) to compare their probabilistic approach for ordering sentences with that of Lapata (2003).
---------------------------------------------------
W06-3407:72	164:220	8 Barzilay and Lee with Exchanges Barzilay and Lee (2004) offer an attractive frame work for constructing a context-specific Hidden Markov Model (HMM) of topic drift.
---------------------------------------------------
W06-3407:73	48:220	Content-oriented models, such as (Barzilay and Lee, 2004), rely on the re-occurrence of patterns of topics over multiple realizations of thematically similar discourses, such as a series of newspaper articles about similar events.
---------------------------------------------------
W06-3407:74	160:220	In addition to possibly narrowing the space of tentative topic-boundaries, exchanges are helpful in that they provide more coarse-grain building blocks for segmentation algorithms that rely on term-distribution as a proxy for dialogue coherence, such as TextTiling (Hearst, 1994, 1997), the Foltz algorithm (Foltz, 1998), Orthonormal Basis (Olney and Cai, 2005), and Barzilay and Lees content modeling approach (Barzilay and Lee, 2004).
---------------------------------------------------
W05-1621:75	27:171	This methodology is very similar to the way [Barzilay and Lee, 2004] evaluate their probabilistic TS model in comparison to the approach of [Lapata, 2003].
---------------------------------------------------
W05-1621:76	7:171	The TS module is hypothesised to simply order a preselected set of information-bearing items such as sentences [Barzilay et al. , 2002; Lapata, 2003; Barzilay and Lee, 2004] or database facts [Dimitromanolaki and Androutsopoulos, 2003; Karamanis et al. , 2004].
---------------------------------------------------
W05-1621:77	9:171	As pointed out by [Karamanis, 2003; Barzilay and Lee, 2004] inter alia, using corpora for automatic evaluation is motivated by the fact that employing human informants in extended psycholinguistic experiments is often simply unfeasible.
---------------------------------------------------
W05-1621:78	43:171	Thus, there might exist more than one equally good solution for TS, a view shared by almost all TS researchers, but which has not been accounted for in the evaluation methodologies of [Karamanis et al. , 2004] and [Barzilay and Lee, 2004].2 Collecting sentence orderings defined by many experts in our domain enables us to investigate the possibility that there might exist many good solutions for TS.
---------------------------------------------------
W06-3309:79	136:153	70 5 Related Work Although not the first to employ a generative approach to directly model content, the seminal work of Barzilay and Lee (2004) is a noteworthy point of reference and comparison.
---------------------------------------------------
W06-3309:80	24:153	(Barzilay and Lee, 2004).
---------------------------------------------------
W06-3309:81	69:153	(2003) and Barzilay and Lee (2004), we employed Hidden Markov Models to model the discourse structure of MEDLINE abstracts.
---------------------------------------------------
W06-3309:82	118:153	4 Discussion An interesting aspect of our generative approach is that we model HMM outputs as Gaussian vectors (log probabilities of observing entire sentences based on our language models), as opposed to sequences of terms, as done in (Barzilay and Lee, 2004).
---------------------------------------------------
P06-1049:83	23:176	2 Related Work Existing methods for sentence ordering are divided into two approaches: making use of chronological information (McKeown et al. , 1999; Lin 385 and Hovy, 2001; Barzilay et al. , 2002; Okazaki et al. , 2004); and learning the natural order of sentences from large corpora not necessarily based on chronological information (Lapata, 2003; Barzilay and Lee, 2004).
---------------------------------------------------
P06-1049:84	36:176	Barzilay and Lee (2004) have proposed content models to deal with topic transition in domain specific text.
---------------------------------------------------
