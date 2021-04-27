N09-2064:1	37:59	(Sagae and Lavie, 2006) extend this method by tuning t on a development set to maximize fscore.2 They populate a chart with constituents whose weight meets the threshold, and use a CKYstyle parsing algorithm to find the heaviest tree, where the weight of a tree is the sum of its constituents weights.
---------------------------------------------------
N09-2064:2	12:59	(Henderson and Brill, 1999) and (Sagae and Lavie, 2006) propose methods for parse hybridization by recombining constituents.
---------------------------------------------------
N09-2064:3	2:59	c 2009 Association for Computational Linguistics Combining Constituent Parsers Victoria Fossum Dept. of Computer Science University of Michigan Ann Arbor, MI 48104 vfossum@umich.edu Kevin Knight Information Sciences Institute University of Southern California Marina del Rey, CA 90292 knight@isi.edu Abstract Combining the 1-best output of multiple parsers via parse selection or parse hybridization improves f-score over the best individual parser (Henderson and Brill, 1999; Sagae and Lavie, 2006).
---------------------------------------------------
N09-2064:4	14:59	First, while constituent recombination (Henderson and Brill, 1999; Sagae and Lavie, 2006) gives a significant improvement in f-score, it tends to flatten the structure of the individual parses.
---------------------------------------------------
P08-1108:5	151:172	5 Related Work Combinations of graph-based and transition-based models for data-driven dependency parsing have previously been explored by Sagae and Lavie (2006), who report improvements of up to 1.7 percentage points over the best single parser when combining three transition-based models and one graph-based model for unlabeled dependency parsing, evaluated on data from the Penn Treebank.
---------------------------------------------------
D07-1096:6	372:410	To combine the outputs of each parser we used the method of Sagae and Lavie (2006).
---------------------------------------------------
D07-1096:7	241:410	It is worth noting that both these systems combine transitionbased base parsers with a graph-based method for parser combination, as first described by Sagae and Lavie (2006).
---------------------------------------------------
W08-2138:8	134:149	However, our simple voting scheme does not guarantee that a well-formed tree is generated, leaving room for further improvements; e.g., as in (Sagae & Lavie, 2006).
---------------------------------------------------
W07-2220:9	29:50	This technique,firstproposedbySagaeandLavie(2006),was used in the highest scoring system in both the multilingual track (Hall et al. , 2007a) and the domain adaptation track (Sagae and Tsujii, 2007).
---------------------------------------------------
D07-1013:10	25:290	For example, Sagae and Lavie (2006) displayed that combining the predictions of both parsing models can lead to significantly improved accuracies.
---------------------------------------------------
D07-1013:11	182:290	It was already known that the two systems make different errors through the work of Sagae and Lavie (2006).
---------------------------------------------------
D07-1013:12	195:290	This technique is similar to the parser voting methods used by Sagae and Lavie (2006).
---------------------------------------------------
D07-1013:13	201:290	Ensemble systems: The error analysis presented in this paper could be used as inspiration for more refined weighting schemes for ensemble systems of the kind proposed by Sagae and Lavie (2006), making the weights depend on a range of linguistic and graph-based factors.
---------------------------------------------------
W07-2217:14	217:259	The best result on this dataset to date (92.7% UAS) is that of Sagae and Lavie (Sagae & Lavie, 2006) who use a parser which combines the predictions of several pre-existing parsers, including McDonalds and Nivres parsers.
---------------------------------------------------
P07-1079:15	116:260	4.3 Determining constraints with dependency parser combination Parser combination has been shown to be a powerful way to obtain very high accuracy in dependency parsing (Sagae and Lavie, 2006).
---------------------------------------------------
P07-1079:16	153:260	To illustrate how this framework allows for improvements in the accuracy of dependency parsing to be used directly to improve the accuracy of HPSG parsing, we showed that by combining the results of different dependency parsers using the search-based parsing ensemble approach of (Sagae and Lavie, 2006), we obtain improved HPSG parsing accuracy as a result of the improved dependency accuracy.
---------------------------------------------------
P07-1079:17	120:260	This has been found to work well in previous work on dependency parser combination (Zeman and Zabokrtsky, 2005; Sagae and Lavie, 2006).
---------------------------------------------------
P07-1079:18	126:260	In the second approach, combination of 3Downloadedfromhttp://sourceforge.net/projects/mstparser the three dependency parsers is done according to the maximum spanning tree combination scheme of Sagae and Lavie (2006), which results in high accuracy of surface dependencies.
---------------------------------------------------
D08-1059:19	168:188	Existing efforts to add search to deterministic parsing include Sagae 569 and Lavie (2006b), which applied best-first search to constituent parsing, and Johansson and Nugues (2006) and Duan et al.
---------------------------------------------------
D08-1059:20	176:188	An existing method to combine multiple parsing algorithms is the ensemble approach (Sagae and Lavie, 2006a), which was reported to be useful in improving dependency parsing (Hall et al., 2007).
---------------------------------------------------
N09-2066:21	6:73	Right-to-left parsing has been used as part of ensemble-based parsers (Sagae & Lavie, 2006; Hall et al., 2007).
---------------------------------------------------
D07-1123:22	20:170	To make better use of the training set, we applied the algorithm in both directions as Johansson and Nugues (2006) and Sagae and Lavie (2006) for all languages except Catalan and Hungarian.
---------------------------------------------------
D07-1097:23	75:119	Following Sagae and Lavie (2006), we let s(a) = summationtextmi=1 wciai, where wci is the average labeled attachment score of parser i for the word class c8 of the dependent of a, and ai is 1 if a  Gi and 0 otherwise.
---------------------------------------------------
D07-1097:24	71:119	3 The Blended Parser The Blended parser is an ensemble system based on the methodology proposed by Sagae and Lavie (2006).
---------------------------------------------------
D07-1111:25	12:130	We provide additional evidence that the parser ensemble approach proposed by Sagae and Lavie (2006a) can be used to improve parsing accuracy, even when only a single parsing algorithm is used, as long as variation can be obtained, for example, by using different learning techniques or changing parsing direction from forward to backward (of course, even greater gains may be achieved when different algorithms are used, although this is not pursued here); and, finally, 4.
---------------------------------------------------
D07-1111:26	11:130	We generalize the standard deterministic stepwise framework to probabilistic parsing, with the use of a best-first search strategy similar to the one employed in constituent parsing by Ratnaparkhi (1997) and later by Sagae and Lavie (2006); 3.
---------------------------------------------------
D07-1111:27	65:130	Sagae and Lavie (2006a) and Zeman and abokrtsk (2005) have observed that reversing the direction of stepwise parsers can be beneficial in parser combinations.
---------------------------------------------------
D07-1111:28	69:130	At test time, each input sentence is parsed using each of the three LR models, and the three resulting dependency structures are combined according to the maximum-spanning-tree parser combination scheme6 (Sagae and Lavie, 2006a) where each dependency proposed by each of the models has the same weight (it is possible that one of the more sophisticated weighting schemes proposed by Sagae and Lavie may be more effective, but these were not attempted).
---------------------------------------------------
D07-1111:29	27:130	Additionally, following Sagae and Lavie (2006), we extend the basic deterministic LR algorithm with a bestfirst search, which results in a parsing strategy similar to generalized LR parsing (Tomita, 1987; 1990), except that we do not perform Tomitas stack-merging operations.
---------------------------------------------------
I08-2097:30	28:160	They used a method of combining several parsers outputs in the framework of MST parsing (Sagae and Lavie, 2006).
---------------------------------------------------
D09-1161:31	13:255	Besides individual parsing models, many system combination methods for parsing have been proposed (Henderson and Brill 1999; Zeman and abokrtsk 2005; Sagae and Lavie 2006) and promising performance improvements have been reported.
---------------------------------------------------
D09-1161:32	38:255	Sagae and Lavie (2006) combine 5 parsers to obtain a score of 92.1, while they report a score of 91.0 for the best single parser in their paper.
---------------------------------------------------
D09-1161:33	91:255	However, as suggested in (Sagae and Lavie 2006), this feature favours precision over recall.
---------------------------------------------------
D09-1161:34	92:255	To solve this issue, Sagae and Lavie (2006) use a threshold to balance them.
---------------------------------------------------
D09-1161:35	244:255	Performance with Charniak parser enhanced by re-ranking plus self-training 5.7 Comparison with Other State-of-the-art Results Table 11 and table 12 compare our method with the other state-of-the-art methods; we use I, B, R, S and C to denote individual model (Charniak 2000; Collins 2000; Bod 2003; Petrov and Klein 2007), bilingual-constrained model (Burkett and Klein 2008) 1 , re-ranking model (Charniak and Johnson 2005, Huang 2008), self-training model (David McClosky 2006) and combination model (Sagae and Lavie 2006) respectively.
---------------------------------------------------
D09-1161:36	248:255	System  F1-Measure I Petrov and Klein (2007) 89.5 Charniak (2000) 89.7 Bod (2003) 90.7 R Collins (2000) 89.7 Charniak and Johnson (2005) 91.4 Huang (2008) 91.7 S David McClosky (2006) 92.1 C Sagae and Lavie (2006) 92.1 Our method 92.6    Table 12.
---------------------------------------------------
D09-1161:37	34:255	Sagae and Lavie (2006) improve this second scheme by introducing a threshold for the constituent count, and search for the tree with the largest number of count from all the possible constituent combination.
---------------------------------------------------
D09-1161:38	86:255	3.3 Constituent Counts Besides the two model scores, we also adopt constituent count as an additional feature inspired by (Henderson and Brill 1999) and (Sagae and Lavie 2006).
---------------------------------------------------
