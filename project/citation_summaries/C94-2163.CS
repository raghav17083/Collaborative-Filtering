W98-1301:1	3:255	Unlike previous treatments of optimality in computational linguistics, starting with EUison (1994), the new approach does not require any explicit marking and counting of constraint violations.
---------------------------------------------------
J96-4003:2	51:561	Ellison (1994), for example, has shown how to map the optimality constraints of Prince and Smolensky (1993) to finite-state automata; given this result, models of 1 Although our assumption of the simultaneous presentation of surface and underlying forms to the learner may seem at first glance to be unnatural as well, it is quite compatible with certain theories of word-based morphology.
---------------------------------------------------
J96-4003:3	33:561	This means that finite-state transducers (FSTs) can be used to represent phonological rules, greatly simplifying the problem of parsing the output of phonological rules in order to obtain the underlying, lexical forms (Koskenniemi 1983; Karttunen 1993; Pulman and Hepple 1993; Bird 1995; Bird and Ellison 1994).
---------------------------------------------------
J96-4003:4	528:561	Nonsegmental approaches include those of Daelemans, Gillis, and Durieux (1994) for learning stress systems, as well as approaches to learning morphology such as Gasser's (1993) system for inducing Semitic morphology, and Ellison's (1992) extensive work on syllabicity, sonority, and harmony.
---------------------------------------------------
J96-4003:5	25:561	Such datadriven models include the stress acquisition models of Daelemans, Gillis, and Durieux (1994) (an application of Instance-based Learning \[Aha, Kibler, and Albert 1991\]) and Gupta and Touretzky (1994) (an application of Error Back-Propagation), as well as Ellison's (1992) Minimum-Description-Length-based model of the acquisition of the basic concepts of syllabicity and the sonority hierarchy.
---------------------------------------------------
J96-4003:6	77:561	More recently, Bird and Ellison (1994) show that a one-level finite-state automaton can model richer phonological structure, such as the multitier representations of autosegmental phonology.
---------------------------------------------------
J98-2006:7	10:181	This shift has, however, had relatively little impact upon computational work (but see Bird and Ellison 1994).
---------------------------------------------------
J98-2006:8	22:181	Ellison (1994) addresses the question of Q 1998 Association for Computational Linguistics Computational Linguistics Volume 24, Number 2 raises a particularly interesting theoretical question in this context: it allows the specification of a ranking among the constraints and allows lower-ranked constraints to be violated in order for higher-ranked constraints to be satisfied.
---------------------------------------------------
J98-2006:9	147:181	This is done using a construction, first suggested in Ellison (1994), which, expressed intuitively, replaces any such constraint function by a finite number of constraint functions having codomain of size two.
---------------------------------------------------
J09-1004:10	47:152	2 Regardless of how optimization is computed, what is relevant for the assessment of theVCDofOTisthedenitionofoptimality.FollowingDenition1,arankingR CON can be seen as a function from candidates to True (if they are optimal) or False (if they are 2GENisrationalifitisrepresentableasanitestatetransducer.Riggles(2004)CONTENDERSalgorithmis anextensionofEllisons(1994)applicationofDijkstras(1959)shortestpathsalgorithmtooptimization inOTthatoperatesovernite-staterepresentationsofGENandE VAL.Ellisonshowedthatifharmonyis usedasthedistancetobeoptimized,thenoptimaloutputscanbeefcientlyfound.TheCONTENDERS algorithmfollowsasimilarstrategybut,insteadofndingtheshortest(i.e.,mostharmonic)pathforone ranking,thealgorithmndsallnon-harmonically-boundedpathsandtherebyoptimizesforallrankings.
---------------------------------------------------
P97-1040:11	83:313	but also stress marks \[x\], feature dpmains such as \[ATRdom\] (Cole L: Kisseberth, 1994) and morphemes such as \[Stem i. All these constituents are formally identicah each marks off an interval on the timeline.
---------------------------------------------------
P97-1040:12	175:313	Timelines like these could be assembled morphologically from one or more lexical entries (Bird & Ellison, 1994), or produced in the course of algorithm (14).
---------------------------------------------------
P97-1040:13	88:313	It is always possible to recover the old representation (4a) from the new one (4b), under the convention that two constituents on the timeline are linked if their interiors overlap (Bird & Ellison, 1994).
---------------------------------------------------
P97-1040:14	308:313	It remains true that for a fixed grammar, the time to do generation is close to linear on the size of the input (Ellison, 1994), which is heartening if we intend to optimize long utterances with respect to a fixed phonology.
---------------------------------------------------
P97-1040:15	310:313	We saw above to set up the representations and constraints efficiently using deterministic finite-state automata, and how to remedy some hidden inefficiencies in the seminal work of (Ellison, 1994), achieving at least a 100-fold observed speedup.
---------------------------------------------------
P97-1040:16	160:313	The idea is essentially due to (Ellison, 1994), and can be boiled down to two lines: (14) Ellison's algorithm (variant).
---------------------------------------------------
P97-1040:17	4:313	In contrast to less restricted theories using Generalized Alignment, OTP's optimal surface forms can be generated with finite-state methods adapted from (Ellison, 1994).
---------------------------------------------------
P02-1008:18	80:231	4 Comprehension in Finite-State OT This section assumes OTs traditional harmony ordering, in which the candidates that survive filtering by Ci are the ones into which Ci inserts fewest ?s. Much computational work on OT has been conducted within a finite-state framework (Ellison, 1994), in keeping with a tradition of finite-state phonology (Johnson, 1972; Kaplan and Kay, 1994).4 4The tradition already included (inviolable) phonological Finite-state OT is a restriction of the formalism discussed above.
---------------------------------------------------
P02-1008:19	86:231	Ellison (1994) exploited such properties to give a production algorithm for finite-state OT.
---------------------------------------------------
P02-1008:20	155:231	(This is to be expected since rewrite-rule 8Ellison (1994) was able to construct PRODUCE(x) from x. One can even build a transducer for PRODUCE that is correct on all inputs that can achieve K violations and returns;on other inputs (signalling that the transducer needs to be recompiled with increased K).
---------------------------------------------------
P02-1008:21	83:231	The other transducers map each y to a single y or z. These finite-state assumptions were proposed (in a different and slightly weaker form) by Ellison (1994).
---------------------------------------------------
P02-1008:22	142:231	T can be applied efficiently to any input string x, whereas Ellison (1994) or Eisner (1997) requires a fresh automaton construction for each x. A nice trick is to build T without 6Pereira and Riley (1997) build a speech recognizer by composing a probabilistic finite-state language model, a finite-state pronouncing dictionary, and a probabilistic finite-state acoustic model.
---------------------------------------------------
C00-1038:23	171:225	Recall standard bestq)aths or shortest-t)aths algorithms that pare a WFSA d(}wn to its 1)aths of minimmn total weight (Dijkstra, 1959; Ellison, 1994).
---------------------------------------------------
C00-1038:24	179:225	By 261 contrast, Ellison's (1994) best-paths construction tbr unbounded constraints, and previously proposed constructions tbr directional-style constraints (set 2.5) only find the optimal outt)ut for a single input, or at best a finite lexicon.
---------------------------------------------------
C00-1038:25	131:225	3.2 Finite-state assumptions We now confine our attention to tinite-state OT grammars, following (Ellison, 1994; Tesar, 1995; Eisner, 1997a; Frank and Satta, 1998; Karttunen, 1998).
---------------------------------------------------
W98-0902:26	33:308	The OT framework itself has been shown to be expressible with weighted finite-state automata, weighted intersection and bestpath algorithms (Ellison 1994) if constraints and OT's GEN component the function from underlying forms to prosodified surface forms are regular sets.
---------------------------------------------------
P96-1049:27	6:86	Several computational methods for OT have been produced within the short amount of time since Prince and Smolensky's paper (Ellison, 1994; Tesar, 1995; Hammond, 1995).
---------------------------------------------------
P96-1049:28	2:86	In this work-in-progress, we build on the work of Ellison (1994) to investigate the possibility of using OT as a parsing tool that derives underlying forms from surface forms.
---------------------------------------------------
P96-1049:29	28:86	V / gru.mad.wet ** gra.dum.wet ** Align,,III~,,,,,L;;;;;;; i,\[:ll * * *!* Figure 2: OT Evaluation for Tagalog Infixation (Morphenm Structure) ( PPWWWWWWW~ I / WPPWWWW~ { WWPP~ (Syllable Structure) { NC00NCONC/\[ | 0NCONCONC~ ~ 00NONCONC} (Phoneme Strtlcturc) x umgradwet I \[ ~ gumradwetl  grumadwet / Candl Cand2 Cand3 Figure 3: Candidate outputs for um+gradwet in an FST 4 Ellison's Conversion Method Ellison (1994) provides a paradigm for converting Optimality Theory constraints into Finite State Transducers.
---------------------------------------------------
W98-0904:30	63:135	This however is not the 9 German orthography is used except for the phonemic writing of voiceless 't') 1Ranking can be implemented along the lines of Ellison (1994).
---------------------------------------------------
W98-0904:31	62:135	11 7 Locality of Constraint Evaluation Ellisons (1994) algorithms offer a way of globally finding optimal candidates out of regular candidate sets.
---------------------------------------------------
W98-0904:32	23:135	The phonological constraint following Ellison (1994) has the form of a regular relation mapping phonological strings into sequences of constraint violation marks ('1') and 'O's which stand for no violation.
---------------------------------------------------
W98-0904:33	133:135	(Ellison, 1994; Eisner, 1997).
---------------------------------------------------
W98-0904:34	36:135	Initial State: A Final State: C Transitions: (i, B, PI/pi), (i, B, HAP/hap), (B, C, OBJ/je),(S, C, OBJ/e), (B, C, IMF/je) Resulting Candidates Initial State: (0, A) Final State: (2, C) Transitions: ((0, A), (1, B) pi), ((1, B), (2, C), e) ((1, B), (2, C), je) Since the candidate set and the constraint in (2') are regular Ellisons (1994) algorithms for getting an automaton containing only optimal candidates, as long as candidate set and evaluation transducer are regular, can be used.
---------------------------------------------------
