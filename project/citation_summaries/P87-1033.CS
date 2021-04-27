C90-3002:1	8:103	Although Kasper (1987) has proposed an algorithm for handling grammars with disjunctive feature specifications, the computational tractability of complex constraints in unification based formalism remains an open problem (Pereira 1987).
---------------------------------------------------
C90-3002:2	7:103	Except for the more simple, although not unproblematic, extensions to PATR-II like the ones proposed by Karttunen (1984), most of these efforts have their root in the work by Rounds, Kasper and Moshier (Rounds & Kasper, 1986; Moshier & Rounds 1987), who give the proof of the existence of a sound, although non classical, logical interPretation for disjunctive and negative feature specifications.
---------------------------------------------------
E89-1029:3	85:99	It is well-known that disjunctive unification is NPcomplete (Kasper 1987).
---------------------------------------------------
J93-4001:4	37:389	There have been several successful efforts in recent years to develop solution algorithms for Boolean combinations of functional constraints that are polynomial for certain special, perhaps typical, cases (Kasper 1987; Maxwell and Kaplan 1989; D6rre and Eisele 1990; Nakano 1991).
---------------------------------------------------
J93-4001:5	165:389	Kasper observed that separating constraints into disjunctive and nondisjunctive parts and processing the nondisjunctive constraints first can improve performance when the nondisjunctive constraints are unsatisfiable (Kasper 1987).
---------------------------------------------------
P98-2154:6	23:145	Moreover, incorporating disjunctive information into internal representation makes parsing more efficient (Kasper, 1987; Eisele and DSrre, 1988; Maxwell and Kaplan, 1991; Hasida, 1986).
---------------------------------------------------
P98-2154:7	84:145	Obtaining compact constraints corresponds to avoiding unnecessary expansions of disjunctions in graph unification (Kasper, 1987; Eisele and DSrre, 1988).
---------------------------------------------------
P90-1025:8	172:343	SFor a full discussion of the issues involved in adding disjunction to complex-fealm'e based formalisms, see Karttamen (1984), Kasper and Rounds (1986), Kasper (1987), and Iohnson (1989).
---------------------------------------------------
C92-4185:9	6:174	Therelore there have been many proposals on how to unify disjunctive feature structures, the most important being Karttunen's (1984) unification with constraints, Kasper's (1987) unification by successive approximation, Eisele & D0rru's (1988) value unification and lately Eisele & D0rre's (1990a, b) unification with named disjunctions.
---------------------------------------------------
C92-4185:10	161:174	Comparing this to the other proposed alternatives we can see that Kasper's (1987) algorithm has a better worst case complexity (2a/2).
---------------------------------------------------
C88-1060:11	225:239	Recent disjunction proposals by/Kasper 1987/and/Eisele and D0rre 1988/are important steps in the development of the necessary computational technology.
---------------------------------------------------
J94-1001:12	392:394	The techniques discussed by Maxwell and Kaplan (1991, 1992) seem most directly compatible with the approach described in this paper, and the methods described by Kasper (1987b), Eisele and D6rre (1988), and Emele (1991) might have important applications as well.
---------------------------------------------------
J94-1001:13	32:394	1 For examples of this approach see Dawar and Vijay-Shanker (1990), D6rre and Eisele (1990), Johnson (1988, 1990a, 1990b, 1991a, 1991b, in press a), Karttunen (1984), Kasper (1987a, 1987b, 1988), Kasper and Rounds (1986, 1990), Langholm (1989), Pereira (1987), and Smolka (1992).
---------------------------------------------------
C90-3013:14	97:187	There is reason to hope that this will often be the case; while disjunction may be widespread in grammar rules and texical entries, Kasper (1987) observes that in his implementation, "in the analysis of a particular sentence most fieatures have a unique value, and some features are not present at, all.
---------------------------------------------------
C90-3013:15	178:187	Unification time here is order n lo,q n in the sizes n of the input structures (Kasper, 1987).
---------------------------------------------------
C90-3013:16	105:187	Kasper (1987) describes a teelmique which, lbr every set of ',~ conjoined disjt,p.ctions, checks the:, con:;~stcncy first of single disjuncls against the delinite part of the description.
---------------------------------------------------
C90-3013:17	8:187	The general problem of unifying two disjunctive feature structures is non-polynomial in the number of disjunctions (Kasper, 1987).
---------------------------------------------------
C90-3013:18	127:187	This variation of the algorithm given in Kasper (1987) is closer t.o Propane's strategy, b~H. the expansion io full \[)N\[," is it.self in general an exponeatia\] pt'ocess and will.
---------------------------------------------------
C90-1013:19	64:83	The unification of disjunctive feature structures is implemented according to Kasper's algorithm (Kasper, 1987).
---------------------------------------------------
J91-4002:20	529:534	In the worst case axioms are interdependent, and general algorithms for the unification of disjunctive constraints, such as the ones of Kasper (1987b) and Eisele and D6rre (1988) are likely to out-perform the techniques described here.
---------------------------------------------------
J91-4002:21	132:534	Kasper (1987a, 1987b) has designed algorithms that involve the encoding of both system networks and the associated realization rules as constraints expressed within a feature logic involving disjunction.
---------------------------------------------------
J91-2001:22	80:322	Various proposals to extend the Kasper-Rounds approach to deal with negative feature values are described by Moshier and Rounds (1987), Moshier (1988), Kasper (1988), Dawar and Vijayashanker (1989, 1990), Langholm (1989); other extensions to this framework are discussed by D6rre and Rounds (1989), Smolka (1988, 1989), and Nebel and Smolka (1989); and Shieber (1989) discusses the integration of such feature systems into a variety of parsing algorithms.
---------------------------------------------------
J91-2001:23	78:322	The most widely known model of feature structures and constraint language is the one developed to explain disjunctive feature values by Kasper and Rounds (1986, 1990) and Kasper (1986, 1987).
---------------------------------------------------
J91-2001:24	126:322	Instead of providing entities in the interpretation that serve as the denotation for "disjunctive" or "negative" features, we follow Kasper and Rounds (1986, 1990), Moshier and Rounds (1987), and Johnson (1988, 1990) in permitting disjunction and negation only in the constraint language.
---------------------------------------------------
J91-2001:25	238:322	As Kasper (1986, 1987) noted in a different setting, the steps of the attribute-value unification algorithm are just applications of the axioms 1-3.
---------------------------------------------------
J91-2001:26	77:322	As explained below, the familiar attribute-value "unification algorithm" can be interpreted as computing the atomic consequences of a purely conjunctive formula (where the graphs it operates on are data structures efficiently representing such formulae), and unification failure corresponds to the unsatisfiability of that conjunction (Kasper and Rounds 1990; Johnson 1988, 1990a; Pereira 1987).
---------------------------------------------------
C92-2076:27	21:221	There is an important body of research concerned with the development of algorithms that avoid the expensive expansion to DNF (e.g. , Kasper, 1987).
---------------------------------------------------
P91-1040:28	13:333	Kasper (1987) and Eisele and DSrre (1988) have tackled this problem and proposed unification methods for disjunctive feature descriptions.
---------------------------------------------------
