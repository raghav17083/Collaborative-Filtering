E09-1091:1	71:186	Generative Transliteration Similarity Model: We also experimented with an extension of Hes W-HMM model (He, 2007).
---------------------------------------------------
D09-1115:2	143:208	(2007a), the features adopted by lattice-based model are arc posterior probability, language model probability, the number of null arcs, the number of hypothesis arcs possessing more than one non-null word and the number of all non-null words.
---------------------------------------------------
D09-1115:3	150:208	ps(arc) is increased by 1110 1/(k+1) if the hypothesis ranking k in the system s contains the arc (Rosti et al., 2007a; He et al., 2008).
---------------------------------------------------
D09-1115:4	153:208	6 Experiments The candidate systems participating in the system combination are as listed in Table 1: System A is a BTG-based system using a MaxEnt-based reordering model; System B is a hierarchical phrase-based system; System C is the Moses decoder (Koehn et al., 2007); System D is a syntax-based system.
---------------------------------------------------
D09-1115:5	8:208	In recent several years, the system combination methods based on confusion networks developed rapidly (Bangalore et al., 2001; Matusov et al., 2006; Sim et al., 2007; Rosti et al., 2007a; Rosti et al., 2007b; Rosti et al., 2008; He et al., 2008), which show state-of-the-art performance in benchmarks.
---------------------------------------------------
D09-1115:6	159:208	(2008), He (2007) and Vogel et al.
---------------------------------------------------
P08-1059:7	21:185	In recent work, Koehn and Hoang (2007) proposed a general framework for including morphological features in a phrase-based SMT system by factoring the representation of words into a vector of morphological features and allowing a phrase-based MT system to work on any of the factored representations, which is implemented in the Moses system.
---------------------------------------------------
P08-1059:8	22:185	Though our motivation is similar to that of Koehn and Hoang (2007), we chose to build an independent component for inflection prediction in isolation rather than folding morphological information into the main translation model.
---------------------------------------------------
P08-1059:9	67:185	First, the corpus is word-aligned using an implementation of lexicalized-HMMs (He, 2007); then the source sentences are parsed into a dependency structure, and the dependency is projected onto the target side following the heuristics described in (Quirk et al., 2005).
---------------------------------------------------
P08-1059:10	32:185	3.1 Lexicon operations For each target language we use a lexicon L which determines the following necessary operations: Stemming: returns the set of possible morphological stems Sw = {s1,,sl} for the word w according to L. 1 Inflection: returns the set of surface word forms Iw = {i1,,im} for the stems Sw according to L. Morphological analysis: returns the set of possible morphological analyses Aw = {a1,,av} for w. A morphological analysis a is a vector of categorical values, where each dimension and its possible values are defined by L. For the morphological analysis operation, we used the same set of morphological features described in (Minkov et al., 2007), that is, seven features for Russian (POS, Person, Number, Gender, Tense, Mood and Case) and 12 for Arabic (POS, Person, Number, Gender, Tense, Mood, Negation, Determiner, Conjunction, Preposition, Object and Possessive pronouns).
---------------------------------------------------
P08-1059:11	14:185	(2007) introduced a way to address these problems by using a rich featurebased model, but did not apply the model to MT. In this paper, we integrate a model that predicts target word inflection in the translations of English into two morphologically complex languages (Russian and Arabic) and show improvements in the MT output.
---------------------------------------------------
P08-1059:12	27:185	Our inflection prediction model is based on (Minkov et al., 2007), who build models to predict the inflected forms of words in Russian and Arabic, but do not apply their work to MT. In contrast, we focus on methods of integration of an inflection prediction model with an MT system, and on evaluation of the models impact on translation.
---------------------------------------------------
P08-1059:13	29:185	3 Inflection prediction models This section describes the task and our model for inflection prediction, following (Minkov et al., 2007).
---------------------------------------------------
P08-1059:14	6:185	1 Introduction One of the outstanding problems for further improving machine translation (MT) systems is the difficulty of dividing the MT problem into sub-problems and tackling each sub-problem in isolation to improve the overall quality of MT. Evidence for this difficulty is the fact that there has been very little work investigating the use of such independent subcomponents, though we started to see some successful cases in the literature, for example in word alignment (Fraser and Marcu, 2007), target language capitalization (Wang et al., 2006) and case marker generation (Toutanova and Suzuki, 2007).
---------------------------------------------------
P08-1059:15	28:185	Other work closely related to ours is (Toutanova and Suzuki, 2007), which uses an independently trained case marker prediction model in an English-Japanese translation system, but it focuses on the problem of generating a small set of closed class words rather than generating inflected forms for each word in translation, and proposes different methods of integration of the components.
---------------------------------------------------
P08-1059:16	182:185	This may be attributed to the fact that human evaluation typically favors the scenario where only word inflections are different (Toutanova and Suzuki, 2007).
---------------------------------------------------
D08-1011:17	158:215	The bilingual translation models used to compute the semantic similarity are from the worddependent HMMs proposed by He (2007), which are trained on two million parallel sentence-pairs selected from the training corpus allowed by the constrained training condition of MT08.
---------------------------------------------------
D08-1011:18	130:215	Similar to (Rosti et al., 2007), each word in the confusion network is associated with a word posterior probability.
---------------------------------------------------
D08-1011:19	104:215	(2007) extended it to a TER-based method for hypothesis alignment.
---------------------------------------------------
D08-1011:20	21:215	2 Confusion-network-based MT system combination The current state-of-the-art is confusion-networkbased MT system combination as described by 98  Rosti and colleagues (Rosti et al., 2007a, Rosti et al., 2007b).
---------------------------------------------------
D08-1011:21	137:215	One is a trigram model estimated from the English side of the parallel training data, and the other is a 5-gram model trained on the English GigaWord corpus from LDC using the MSRLM toolkit (Nguyen et al, 2007).
---------------------------------------------------
