P09-1030:1	237:255	There is some recentworkoncompilingdictionariesfrommonolingual corpora, which may scale to several language pairs in future (Haghighi et al., 2008).
---------------------------------------------------
W09-1117:2	31:164	Haghighi et al., (2008) made use of contextual and orthographic clues for learning a generative model from monolingual corpora and a seed lexicon.
---------------------------------------------------
W09-1117:3	20:164	2 Related Work The literature on translation lexicon induction for low-density languages falls in to two broad categories: 1) Effectively utilizing similarity between languages by choosing a high-resource bridge language for translation (Mann and Yarowsky, 2001; Schafer and Yarowsky, 2002) and 2) Extracting noisy clues (such as similar context) from monolingual corpora with help of a seed lexicon (Rapp, 1999; Koehn and Knight, 2002; Schafer and Yarowsky, 2002, Haghighi et al., 2008).
---------------------------------------------------
W09-1117:4	19:164	129 We further show that an extension based on partof-speech clustering can give similar accuracy gains for learning translations of all word-types, deepening the findings of previous literature which mainly focused on translating nouns (Rapp, 1999; Koehn and Knight, 2002; Haghighi et al., 2008).
---------------------------------------------------
W09-1117:5	10:164	The marked difference in the availability of monolingual vs parallel corpora has led several researchers to develop methods for automatically learning bilingual lexicons, either by using monolingual corpora (Rapp, 1999; Koehn and Knight, 2002; Schafer and Yarowsky, 2002; Haghighi et al., 2008) or by exploiting the cross-language evidence of closely related bridge languages that have more resources (Mann and Yarowsky, 2001).
---------------------------------------------------
W09-1117:6	60:164	3.1.1 Baseline model In the baseline model, the context is computed using adjacent words as in (Rapp,1999; Koehn and Knight, 2002; Schafer and Yarowsky, 2002; Haghighi et al., 2008).
---------------------------------------------------
W09-1117:7	101:164	Because it would be intractable to compare each projected vector against the vectors for all possible English words, we limited ourselves to comparing the projected vector from each Spanish word against the vectors for the 1000 most frequent English nouns, following along the lines of previous work (Koehn and Knight, 2002; Haghighi et al., 2008).
---------------------------------------------------
W09-1117:8	124:164	6 Further Extensions: Generalizing to other word types via tagset mapping Most of the previous literature on this problem focuses on evaluating on nouns (Rapp, 1999; Koehn and Knight 2002; Haghighi et al., 2008).
---------------------------------------------------
W09-1702:9	93:146	This approach is a simple alternative to replace the 10-20k general dictionaries (Rapp, 1999; Fung and McKeown, 2004) or automatic seed words (Koehn and Knight, 2002; Haghighi et al., 2008).
---------------------------------------------------
W09-1702:10	39:146	Haghighi et al (2008) only use a small-sized bilingual lexicon containing 100 word pairs as seed lexicon.
---------------------------------------------------
W09-1702:11	137:146	Haghighi et al (2008) have reported that the most common errors detected in their analysis on top 100 errors were from semantically related words, which had strong context feature correlations.
---------------------------------------------------
D09-1092:12	147:231	We therefore evaluate the ability of the PLTM to generate bilingual lexica, similar to other work in unsupervised translation modeling (Haghighi et al., 2008).
---------------------------------------------------
