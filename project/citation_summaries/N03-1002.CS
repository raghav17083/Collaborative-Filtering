I08-2080:1	7:170	Therefore, Japanese NER has tight relation with morphological analysis, and thus it is often performed immediately after morphological analysis (Masayuki and Matsumoto, 2003; Yamada, 2007).
---------------------------------------------------
I08-2080:2	147:170	Use all structural information and thesaurus Since(MasayukiandMatsumoto,2003; Nakanoand Hirai, 2004) reported the performance of NER system was improved by using a thesaurus, we also conducted experiment in which semantic classes obtained from a Japanese thesaurus  Bunrui Goi Hyo (NLRI, 1993) were added to the SVM features.
---------------------------------------------------
I08-2080:3	48:170	4 NER Using Structural Information 4.1 Outline of Our NER System Our NER system performs the chunking process based on morpheme units because character-based methods do not outperform morpheme-based methods (Masayuki and Matsumoto, 2003) and are not suitable for considering wider context.
---------------------------------------------------
I08-2080:4	161:170	CRL cross IREX Learning Analysis Featuresvalidation test data Method Units (Isozaki and Kazawa, 2003) 86.77 85.10 SVM + Viterbi morpheme basic features (Masayuki and Matsumoto, 2003) 87.21 SVM character +thesaurus (Fukuoka, 2006) 87.71 Semi-Markov CRF character basic features (Yamada, 2007) 88.33 SVM + Shift-Reduce morpheme +bunsetsu features (Nakano and Hirai, 2004) 89.03 SVM character +bunsetsu features & thesaurus Our system 89.40 87.72 SVM + Viterbi morpheme +structural information & thesaurus IREX test data.
---------------------------------------------------
I08-5008:5	19:195	The other SVM-based NER systems can be found in (Takeuchi and Collier, 2002) and (Asahara and Matsumoto, 2003).
---------------------------------------------------
