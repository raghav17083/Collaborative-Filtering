W97-1013:1	253:294	tor sentence length >2 \[ _>7 \[ _>10 3.98 (11.29) 13.55 q22.84) 37.46 t41.35) 1.28 (2.31) 2.98,4.46) 6.2118.67) 1.33 (2.43) 3.18~ 4.69) 6.8518.97) number of nodes in trees 141960 (938) 117134 (1627) 818.5 (10.26) 117750 (1551) 812 (lO.4O) Table 2: Means and STDs of ten experiments (OVIS), ParPar denotes Partial-Parser employs prior to disambiguation (Sima'an, 1996a), and for specifying the cut-nodes for DOP.
---------------------------------------------------
W97-1013:2	138:294	The partial-parser is implemented as a parser for TSGs (Sima'an, 1996a), based on an extension to the CYK algorithm (Younger, 1967)).
---------------------------------------------------
W97-1013:3	248:294	(Sima'an, 1996a)): for each projected partial-tree, a maximum was set on its depth (D), number of substitution-sites (N) on its frontier, number of words (W) and number of consecutive words (C) on its frontier.
---------------------------------------------------
W97-1013:4	13:294	For many applications (e.g. Speech Understanding), probabilistic evaluation of the full parse-space using such models is NPhard (Sima'an, 1996b), and even when it is deterministic polynomial-time, then grammar size is prohibitive.
---------------------------------------------------
P98-1021:5	25:195	The most probable parse can be estimated by iterative Monte Carlo sampling (Bod 1995), but efficient algorithms exist only for sub-optimal solutions such as the most likely derivation of a sentence (Bod 1995, Sima'an 1995) or the "labelled recall parse" of a sentence (Goodman 1996).
---------------------------------------------------
P98-1021:6	24:195	However, the computation of the most probable parse of a sentence is NP-hard (Sima'an 1996).
---------------------------------------------------
C00-1011:7	75:160	Although Goodman's rcductkm method does still not allow for an efficient computation {51 tile most probable parse in DOP (ill fact, the prol~lem of computing the most prolmble parse is NP-hard -sue Sima'an 1996), his method does allow for an efficient computation o1' the "nmximun~ constituents parse", i.e., the parse tree that is most likely to have the largest number of correct constitueuts (also called the "labeled recall parse").
---------------------------------------------------
W06-2912:8	96:174	In fact, the problem of computing the most probable tree in DOP is known to be NP hard (Sima'an 1996).
---------------------------------------------------
C00-2092:9	93:175	Furthermore, it has been shown that the Viterbi algorithm cannot be used to make the most probable selection from a DOP-like derivation forest (Sima'an, 1996).
---------------------------------------------------
C00-2099:10	81:179	Otherwise, the result will be approxima.ting the parse probabi\]ity with a derivation probability, as described in detail in (Samuelsson, 2000) based on the seminal work of (Sima'an, 1996).
---------------------------------------------------
P06-1109:11	60:166	Since the computation of the most probable parse tree is NP-complete (Sima'an 1996), U-DOP estimates the most probable tree from the 100 most probable derivations using Viterbi n-best parsing.
---------------------------------------------------
P97-1021:12	107:230	As was shown in Sima'an (1996b), the most likely interpretation of a string cannot be computed in deterministic polynomial time.
---------------------------------------------------
P97-1021:13	45:230	Bod (1996), Sima'an (1996a), Goodman (1996)).
---------------------------------------------------
P97-1021:14	21:230	For the syntactic dimension of language, various instantiations of this data-oriented processing or "DOP" approach have been worked out (e.g. Bod (1992-1995); Charniak (1996); Tugwell (1995); Sima'an et al.
---------------------------------------------------
P97-1021:15	115:230	The current implementation is again an extension of Sima'an (1996a), by Bonnema 2.
---------------------------------------------------
P97-1021:16	105:230	The probability of an interpretation I of a string is the sum of the probabilities of the parses of this string with a top node annotated with a formula that is provably equivalent to I. Let ti4p be the i-th subtree in the derivation d that yields parse p with interpretation I, then the probability of I is given by: P(I) = E E H P(t,d,) (4) p d i We choose the most probable interpretation/.of a string s as the most appropriate interpretation of s. In Bonnema (1996) a semantic extension of the DOP parser of Sima'an (1996a) is given.
---------------------------------------------------
P97-1021:17	22:230	(1994); Sima'an (1994; 1996a); Goodman (1996); Rajman (1995ab); Kaplan (1996); Sekine and Grishman (1995)).
---------------------------------------------------
