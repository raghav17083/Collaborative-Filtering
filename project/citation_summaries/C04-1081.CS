I08-1044:1	37:215	CRFs have been successfully applied to a number of real-world tasks, including NP chunking (Sha and Pereira, 2003), Chinese word segmentation (Peng et al., 2004), information extraction (Pinto et al., 2003; Peng and McCallum, 2004), named entity identification (McCallum and Li, 2003; Settles, 2004), and many others.
---------------------------------------------------
N06-2049:2	111:132	Later, this approach was implemented by the CRF-based method (Peng and McCallum, 2004), which was proved to achieve better results than the maximum entropy approach because it can solve the label bias problem (Lafferty et al. , 2001).
---------------------------------------------------
N06-2049:3	6:132	1 Introduction The character-based IOB tagging approach has been widely used in Chinese word segmentation recently (Xue and Shen, 2003; Peng and McCallum, 2004; Tseng et al. , 2005).
---------------------------------------------------
I08-4015:4	5:83	1 Introduction Since Chinese Word Segmentation was firstly treated as a character-based tagging task in (Xue and Converse, 2002), this method has been widely accepted and further developed by researchers (Peng et al., 2004), (Tseng et al., 2005), (Low et al., 2005), (Zhao et al., 2006).
---------------------------------------------------
I05-3027:5	49:78	(2004) and our Fscore on the closed track in Sighan bakeoff 2003 Sighan Bakeoff 2003 Our F-score F-score Peng et al.
---------------------------------------------------
W08-0336:6	12:196	Chinese information retrieval (IR) systems benefit from a segmentation that breaks compound words into shorter words (Peng et al., 2002), paralleling the IR gains from compound splitting in languages like German (Hollink et al., 2004), whereas automatic speech recognition (ASR) systems prefer having longer words in the speech lexicon (Gao et al., 2005).
---------------------------------------------------
W06-0140:7	29:99	Different from (Peng et al. , 2004), we represent the positions of a hanzi (Chinese character) with four different tags: B for a hanzi that starts a word, I for a hanzi that continues the word, F for a hanzi that ends the word, S for a hanzi that occurs as a single-character word.
---------------------------------------------------
I08-4033:8	37:113	Before the subword-based tagging, the character-based IOB tagging approach has been widely used in Chinese word segmentation recently (Xue and Shen, 2003; Peng and McCallum, 2004; Tseng et al., 2005).
---------------------------------------------------
I08-4033:9	13:113	Segmentation performance has been improved significantly, from the earliest maximal match (dictionary-based) approaches to HMM-based (Zhang et al., 2003) approaches and recent state-of-the-art machine learning approaches such as maximum entropy (MaxEnt) (Xue and Shen, 2003), support vector machine (SVM) (Kudo and Matsumoto, 2001), conditional random fields (CRF) (Peng and McCallum, 2004), and minimum error rate training (Gao et al., 2004).
---------------------------------------------------
I08-4009:10	16:159	Since CWS was firstly treated as a characterbased tagging task (we call it CT for short hereafter) in (Xue and Converse, 2002), this method has been widely accepted and further developed by researchers (Peng et al., 2004), (Tseng et al., 2005), (Low et al., 2005), (Zhao et al., 2006).
---------------------------------------------------
W06-0139:11	14:74	Among them, the machine learning-based techniques showed excellent performance in many research studies (Peng et al. , 2004; Zhou et al. , 2005; Gao et al. , 2004).
---------------------------------------------------
C08-1113:12	107:177	We also used lexical features consulting a dictionary: one is to check if any of the above defined character n-grams appear in a dictionary (Peng et al., 2004), and the other is to check if there are any words in the dictionary that start or end at the current character boundary.
---------------------------------------------------
P07-1106:13	102:195	It can be seen as an alternative to the ME and CRF models (Xue, 2003; Peng et al. , 2004), which 843 1 word w 2 word bigram w1w2 3 single-character word w 4 a word starting with character c and having length l 5 a word ending with character c and having length l 6 space-separated characters c1 and c2 7 character bigram c1c2 in any word 8 the first and last characters c1 and c2 of any word 9 word w immediately before character c 10 character c immediately before word w 11 the starting characters c1 and c2 of two consecutive words 12 the ending characters c1 and c2 of two consecutive words 13 a word of length l and the previous word w 14 a word of length l and the next word w Table 1: feature templates do not involve word information.
---------------------------------------------------
W06-0130:14	18:90	The superiority of CRFs on Chinese information processing was also demonstrated in word segmentation (Peng et al. 2004).
---------------------------------------------------
H05-1060:15	87:281	When training a CRF for POS tagging, IOB chunking (Sha and Pereira, 2003), or word segmentation (Peng et al. , 2004), one typically structures the conditional probabilities (in the objective function) using domain knowledge: in POS tagging, the set of allowed tags for a word is used; in IOB chunking, the bigram O I is disallowed; and in segmentation, a lexicon is used to enumerate the possible word boundaries.4 4This refinement is in the same vein as the move from maximum likelihood estimation to conditional estimation.
---------------------------------------------------
P06-1126:16	8:237	1 Introduction Chinese word segmentation is the initial stage of many Chinese language processing tasks, and has received a lot of attention in the literature (Sproat et al. , 1996; Sun and Tsou, 2001; Zhang et al. , 2003; Peng et al. , 2004).
---------------------------------------------------
I08-4023:17	9:62	Among them,the machine learning-based techniques showed excellent performance inmany researchstudies (Peng et al., 2004; Zhou et al., 2005; Gao et al., 2004).
---------------------------------------------------
P06-2123:18	11:217	(SVM) (Kudo and Matsumoto, 2001), conditional random fields (CRF) (Peng and McCallum, 2004), and minimum error rate training (Gao et al. , 2004).
---------------------------------------------------
P06-2123:19	187:217	Later, (Peng and McCallum, 2004) implemented the idea using the CRF-based approach, which yielded better results than the maximum entropy approach because it could solve the label bias problem (Lafferty et al. , 2001).
---------------------------------------------------
J05-4005:20	787:855	# OAS LN PN ON Segmenters errors P R F P R F P R F MSWS 63 .935 .442 .600 .907 .744 .818 .642 .469 .600 LCWS 49 .854 .720 .782 .945 .781 .856 .713 .131 .222 PBWS 20 .767 .736 .752 .780 .787 .784 .817 .216 .342 MSRSeg 7 .876 .864 .870 .830 .897 .862 .799 .617 .696 Table 23 Comparisons against other segmenters: In Column 1, SXX indicates participating sites in the 1 st SIGHAN International Chinese Word Segmentation Bakeoff, and CRFs indicates the word segmenter reported in (Peng et al. 2004).
---------------------------------------------------
I08-4029:21	15:118	Among them, the machine learning-based techniques showed excellent performance in many recent research studies (Peng et al., 2004; Zhou et al., 2005; Gao et al., 2004).
---------------------------------------------------
I05-3035:22	18:94	Different from (Peng et al. , 2004), we represent the positions of a hanzi (Chinese character) with four different tags: B for a hanzi 196 that starts a word, I for a hanzi that continues the word, F for a hanzi that ends the word, S for a hanzi that occurs as a single-character word.
---------------------------------------------------
