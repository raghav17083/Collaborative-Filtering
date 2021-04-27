D09-1134:1	53:207	Another approach (Suzuki and Isozaki, 2008) embeds a joint probability model (HMM in their 1We slightly modify the notation here to be consistent with the rest of the paper.
---------------------------------------------------
E09-1087:2	7:152	Most recently, (Suzuki and Isozaki, 2008) published their Semi-supervised sequential labelling method, whose results on POS tagging seem to be optically better than (Shen et al., 2007), but no significance tests were given and the tool is not available for download, i.e. for repeating the results and significance testing.
---------------------------------------------------
N09-1059:3	31:202	Suzuki and Isozaki (2008) provided evidence that the use of more unlabeled data in semisupervised learning could improve the performance of NLP tasks, such as POS tagging, syntactic chunking, and named entities recognition.
---------------------------------------------------
D09-1058:4	85:216	Note that it is possible to iterate the methodsteps 2 and 3 can be repeated multiple times (Suzuki and Isozaki, 2008)but in our experiments we only performed these steps once.
---------------------------------------------------
D09-1058:5	210:216	7 Conclusion This paper has described an extension of the semi-supervised learning approach of (Suzuki and Isozaki, 2008) to the dependency parsing problem.
---------------------------------------------------
D09-1058:6	140:216	We follow a similar approach to that of (Suzuki and Isozaki, 2008) in partitioning f(x,y), where the k different feature vectors correspond to different feature types or feature templates.
---------------------------------------------------
D09-1058:7	15:216	Our approach basically follows a framework proposed in (Suzuki and Isozaki, 2008).
---------------------------------------------------
D09-1058:8	3:216	We describe an extension of semisupervised structured conditional models (SS-SCMs) to the dependency parsing problem, whose framework is originally proposed in (Suzuki and Isozaki, 2008).
---------------------------------------------------
W09-1119:9	139:223	6.1 Unlabeled Text Recent successful semi-supervised systems (Ando and Zhang, 2005; Suzuki and Isozaki, 2008) have illustrated that unlabeled text can be used to improve the performance of NER systems.
---------------------------------------------------
W09-1119:10	202:223	NER proves to be a knowledgeintensive task, and it was reassuring to observe that System Resources Used F1 + LBJ-NER Wikipedia, Nonlocal Features, Word-class Model 90.80 (Suzuki and Isozaki, 2008) Semi-supervised on 1Gword unlabeled data 89.92 (Ando and Zhang, 2005) Semi-supervised on 27Mword unlabeled data 89.31 (Kazama and Torisawa, 2007a) Wikipedia 88.02 (Krishnan and Manning, 2006) Non-local Features 87.24 (Kazama and Torisawa, 2007b) Non-local Features 87.17 + (Finkel et al., 2005) Non-local Features 86.86 Table 7: Results for CoNLL03 data reported in the literature.
---------------------------------------------------
P09-1081:11	16:188	Recent research indicates that using labeled and unlabeled data in semi-supervised learning (SSL) environment, with an emphasis on graph-based methods, can improve the performance of information extraction from data for tasks such as question classification (Tri et al., 2006), web classification (Liu et al., 2006), relation extraction (Chen et al., 2006), passage-retrieval (Otterbacher et al., 2009), various natural language processing tasks such as partof-speech tagging, and named-entity recognition (Suzuki and Isozaki, 2008), word-sense disam719 biguation (Niu et al., 2005), etc. We consider situations where there are much more unlabeled data, XU, than labeled data, XL, i.e., nL lessmuch nU.
---------------------------------------------------
P09-1116:12	134:254	In comparison, there are 79 templates in (Suzuki and Isozaki, 2008).
---------------------------------------------------
P09-1116:13	238:254	Although the method in (Suzuki and Isozaki 2008) is quite general, it is hard to see how it can be applied to the query classification problem.
---------------------------------------------------
P09-1116:14	225:254	al. 2004), (Wong and Ng 2007), (Suzuki and Isozaki 2008), and (Koo et.
---------------------------------------------------
P09-1116:15	227:254	Wong and Ng (2007) and Suzuki and Isozaki (2008) are similar in that they run a baseline discriminative classifier on unlabeled data to generate pseudo examples, which are then used to train a different type of classifier for the same problem.
---------------------------------------------------
P09-1116:16	190:254	3.1) 83.78 W500 88.34 +4.56 P64 9.73 +5.94 P125 89.80 +6.02 W500 + P125 90.62 +6.84 W500 + P64 0.63 +6.85 W500 + P125 + P64 90.90 +7.12 W500 + P125 + P64+pos 90.62 +6.84 LDC64 87.24 +3.46 LDC125 8.33 +4.55 LDC64 +LDC125 88.44 +4.66 (Suzuki and Isozaki, 2008) 89.92 (Ando and Zhang, 2005) 89.31 (Florian et al., 2003) 88.76 (Chieu and Ng, 2003) 88.31 (Klein et al., 2003) 86.31 Table 5 Example queries and their classes ford field    Sports/American Football    Information/Local & Regional    Sports/Schedules & Tickets john deere gator    Living/Landscaping & Gardening    Living/Tools & Hardware    Information/Companies & Industries    Shopping/Stores & Products    Shopping/Buying Guides & Researching justin timberlake lyrics    Entertainment/Music    Information/Arts & Humanities    Entertainment/Celebrities Table 6 Labeler Consistency  L1  L2 L3 Average F1 0.538 0.477 0.512 0.509 P 0.501 0.613 0.463 0.526 1035 Given an input x, represented as a vector of m features: (x1, x2, , xm), a logistic regression classifier with parameter vector L(w1, w2, , wm) computes the posterior probability of the output y, which is either 1 or -1, as L:U
---------------------------------------------------
P09-1116:17	229:254	Suzuki and Isozaki (2008), on the other hand, used the automatically labeled corpus to train HMMs.
---------------------------------------------------
P09-1116:18	135:254	Part-of-speech tags were used in the topranked systems in CoNLL 2003, as well as in many follow up studies that used the data set (Ando and Zhang 2005; Suzuki and Isozaki 2008).
---------------------------------------------------
