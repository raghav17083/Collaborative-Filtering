W06-3119:1	7:125	1 Introduction Recent work in machine translation has evolved from the traditional word (Brown et al. , 1993) and phrase based (Koehn et al. , 2003a) models to include hierarchical phrase models (Chiang, 2005) and bilingual synchronous grammars (Melamed, 2004).
---------------------------------------------------
D07-1003:2	39:243	2 Related Work To model the syntactic transformation process, researchers in these fieldsespecially in machine translationhave developed powerful grammatical formalisms and statistical models for representing and learning these tree-to-tree relations (Wu and Wong, 1998; Eisner, 2003; Gildea, 2003; Melamed, 2004; Ding and Palmer, 2005; Quirk et al. , 2005; Galley et al. , 2006; Smith and Eisner, 2006, inter alia).
---------------------------------------------------
P09-1090:3	29:153	To handle syntactic differences, Melamed (2004) proposes methods based on tree-to-tree mappings.
---------------------------------------------------
P07-1108:4	8:179	1 Introduction For statistical machine translation (SMT), phrasebased methods (Koehn et al. , 2003; Och and Ney, 2004) and syntax-based methods (Wu, 1997; Alshawi et al. 2000; Yamada and Knignt, 2001; Melamed, 2004; Chiang, 2005; Quick et al. , 2005; Mellebeek et al. , 2006) outperform word-based methods (Brown et al. , 1993).
---------------------------------------------------
D07-1091:5	32:185	The goal of integrating syntactic information into the translation model has prompted many researchers to pursue tree-based transfer models (Wu, 1997; Alshawi et al. , 1998; Yamada and Knight, 2001; Melamed, 2004; Menezes and Quirk, 2005; Galley et al. , 2006), with increasingly encouraging results.
---------------------------------------------------
W06-1628:6	66:291	Melamed (2004) establishes a theoretical framework for generalized synchronous parsing and translation.
---------------------------------------------------
D07-1008:7	41:382	Proposals include Eisners (2003) synchronous tree substitution grammar (STSG), Melameds (2004) multitext grammar, and Graehl and Knights (2004) tree-to-tree transducers.
---------------------------------------------------
C08-1138:8	120:198	Our synchronous parser is similar to the synchronous CKY parser presented at (Melamed, 2004).
---------------------------------------------------
C08-1138:9	116:198	When a parsers grammar can have fewer dimensions than the parsers input, we call it a synchronizer (Melamed, 2004).
---------------------------------------------------
P07-1020:10	182:189	The BOW approach is different from the parsing based approaches (Melamed, 2004; Zhang and Gildea, 2005; Cowan et al. , 2006) where the translation model tightly couples the syntactic and lexical items of the two languages.
---------------------------------------------------
W06-3104:11	34:270	It differs from the many approaches where (1) is defined by a stochastic synchronous grammar (Wu, 1997; Alshawi et al. , 2000; Yamada and Knight, 2001; Eisner, 2003; Gildea, 2003; Melamed, 2004) and from transfer-based systems defined by context-free grammars (Lavie et al. , 2003).
---------------------------------------------------
W06-3104:12	46:270	However, one would need rather powerful and slow grammar formalisms (Shieber and Schabes, 1990; Melamed et al. , 2004), often with discontiguous constituents, to account for all the linguistic divergences that could arise from different movement patterns (scrambling, wh-in situ) or free translation.
---------------------------------------------------
N06-1031:13	6:157	Some approaches have used syntax at the core (Wu, 1997; Alshawi et al. , 2000; Yamada and Knight, 2001; Gildea, 2003; Eisner, 2003; Hearne and Way, 2003; Melamed, 2004) while others have integrated syntax into existing phrase-based frameworks (Xia and McCord, 2004; Chiang, 2005; Collins et al. , 2005; Quirk et al. , 2005).
---------------------------------------------------
D09-1021:14	28:224	Early examples of this work include (Alshawi, 1996; Wu, 1997); more recent models include (Yamada and Knight, 2001; Eisner, 2003; Melamed, 2004; Zhang and Gildea, 2005; Chiang, 2005; Quirk et al., 2005; Marcu et al., 2006; Zollmann and Venugopal, 2006; Nesson et al., 2006; Cherry, 2008; Mi et al., 2008; Shen et al., 2008).
---------------------------------------------------
W05-0803:15	13:160	The only requirement will be that a parallel corpus exist for the language under consideration and one or more other languages.2 Induction of grammars from parallel corpora is rarely viewed as a promising task in its own right; in work that has addressed the issue directly (Wu, 1997; Melamed, 2003; Melamed, 2004), the synchronous grammar is mainly viewed as instrumental in the process of improving the translation model in a noisy channel approach to statistical MT.3 In the present paper, we provide an important prerequisite for parallel corpus-based grammar induction work: an efficient algorithm for synchronous parsing of sentence pairs, given a word alignment.
---------------------------------------------------
P06-1077:16	17:252	Melamed (2004) formalizes machine translation problem as synchronous parsing based on multitext grammars.
---------------------------------------------------
W07-0401:17	12:352	(Wu, 1997; Yamada and Knight, 2001; Gildea, 2003; Melamed, 2004; Graehl and Knight, 2004; Galley et al. , 2006).
---------------------------------------------------
W06-3106:18	14:161	(Melamed, 2004; Ding and Palmer, 2005)) and several algorithms have been proposed to learn the underlying production rule probabilities (Graehl and Knight, 2004; Ding and Palmer, 2004).
---------------------------------------------------
E09-1049:19	16:164	A representative sample of modern syntax-based systems includes models based on bilingual synchronous grammar (Melamed, 2004), parse tree-to-string translation models (Yamada and Knight, 2001) and nonisomorphic tree-to-tree mappings (Eisner, 2003).
---------------------------------------------------
I08-1067:20	32:124	Graehl and Knight (2004) and Melamed (2004), propose methods based on tree-to-tree mappings.
---------------------------------------------------
H05-1101:21	58:170	The result could be derived from the findings in (Melamed et al. , 2004) that synchronous rewriting systemsasSCFGsarerelatedtotheclassofsocalled linear context-free rewriting systems (LCFRSs) and from the result that the membership problem for 805 LCFRSs is NP-hard (Satta, 1992; Kaji and others, 1994).
---------------------------------------------------
H05-1101:22	132:170	This has been shown for instance in (Wu and Wong, 1998; Yamada and Knight, 2001; Melamed, 2004).
---------------------------------------------------
P05-1066:23	13:229	For this reason there is currently a great deal of interest in methods which incorporate syntactic information within statistical machine translation systems (e.g. , see (Alshawi, 1996; Wu, 1997; Yamada and Knight, 2001; Gildea, 2003; Melamed, 2004; Graehl and Knight, 2004; Och et al. , 2004; Xia and McCord, 2004)).
---------------------------------------------------
P05-1066:24	41:229	2.1.2 Research on Syntax-Based SMT A number of researchers (Alshawi, 1996; Wu, 1997; Yamada and Knight, 2001; Gildea, 2003; Melamed, 2004; Graehl and Knight, 2004; Galley et al. , 2004) have proposed models where the translation process involves syntactic representations of the source and/or target languages.
---------------------------------------------------
N06-1056:25	46:173	Synchronous parsing has seen a surge of interest recently in the machine translation community as a way of formalizing syntax-based translation models (Melamed, 2004; Chiang, 2005).
---------------------------------------------------
P09-1053:26	48:248	These constraints imply permissible configurations. For example, requiringlnegationslash= 0 and, if knegationslash= 0 then sk must be a child of sl in the source tree, we can implement a synchronous dependency grammar similar to (Melamed, 2004).
---------------------------------------------------
P05-1067:27	24:217	Melamed (2003, 2004) formalized the MT problem as synchronous parsing based on multitext grammars.
---------------------------------------------------
D08-1012:28	101:215	See Wu (1996) or Melamed (2004) for a detailed exposition.
---------------------------------------------------
W07-0706:29	233:240	And finally, our model allow gaps on the target side, which is an advantage shared by (Melamed, 2004) and (Simard, 2005).
---------------------------------------------------
W07-0706:30	147:240	Melamed (2004) also used a similar way to integrate the language model.
---------------------------------------------------
D07-1079:31	10:289	By contrast, explicit syntax approaches seek to directly model the relations learned from parsed data, including models between source trees and target trees (Gildea, 2003; Eisner, 2003; Melamed, 2004; Cowan et al. , 2006), source trees and target strings (Quirk et al. , 2005; Huang et al. , 2006), or source strings and target trees (Yamada and Knight, 2001; Galley et al. , 2004).
---------------------------------------------------
P04-1084:32	23:180	Next, we prove that in GMTG each component grammar retains its generative power, a requirement for synchronous formalisms that Rambow and Satta (1996) called the weak language preservation property. Lastly, we propose a synchronous generalization of Chomsky Normal Form, which lays the groundwork for synchronous parsing under GMTG using a CKYstyle algorithm (Younger, 1967; Melamed, 2004).
---------------------------------------------------
P04-1084:33	155:180	The absence of a22 s simplifies parsers for GMTG (Melamed, 2004).
---------------------------------------------------
E09-1061:34	92:222	Syntax-based translation logics are similar to parsing logics; a few examples already appear in the literature (Chiang, 2007; Venugopal et al., 2007; Dyer et al., 2008; Melamed, 2004).
---------------------------------------------------
E09-1061:35	12:222	It is occasionally used to describe formally syntactic translation models, but these treatments tend to be brief (Chiang, 2007; Venugopal et al., 2007; Dyer et al., 2008; Melamed, 2004).
---------------------------------------------------
