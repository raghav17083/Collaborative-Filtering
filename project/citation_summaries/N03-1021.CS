P09-1009:1	43:226	More general formalisms for the same purpose were later developed (Wu and Wong, 1998; Chiang, 2005; Melamed, 2003; Eisner, 2003; Zhang and Gildea, 2005; Blunsom et al., 2008).
---------------------------------------------------
W07-0405:2	35:147	As a result, the binarized grammar, with a gap on the target-side, is no longer an SCFG, but can be represented in the more general formalism of Multi-Text Grammars (MTG) (Melamed, 2003): (3) parenleftbiggS S parenrightbigg  [1,2][1,2,1] parenleftbigg NP-PP VP NP-PP (2) VP parenrightbigg here [1,2,1] denotes that on that target-side, the first nonterminal NP-PP has two discontinuous spans, with the second nonterminal VP in the gap.
---------------------------------------------------
P05-1059:3	10:146	Melamed (2003) presents algorithms for synchronous parsing with more complex grammars, discussing how to parse grammars with greater than binary branching and lexicalization of synchronous grammars.
---------------------------------------------------
P04-1083:4	64:203	(2000), and Melamed (2003), because it uses only one kind of item, and it never composes terminals.
---------------------------------------------------
P04-1083:5	23:203	This normal form allows simpler algorithm descriptions than the normal forms used by Wu (1997) and Melamed (2003).
---------------------------------------------------
P04-1083:6	72:203	Under such an MTG, the logic of word alignment is the one in Melamed (2003)s Parser A, but without Compose inferences.
---------------------------------------------------
P04-1083:7	53:203	3 A Synchronous CKY Parser Figure 3 shows Logic C. Parser C is any parser based on Logic C. As in Melamed (2003)s Parser A, Parser Cs items consist of a a0 -dimensional label vector a2a50a49a51 and a a0 -dimensional d-span vector a52 a49 a51 . 2 The items contain d-spans, rather than ordinary spans, because 2Superscripts and subscripts indicate the range of dimensions of a vector.
---------------------------------------------------
P04-1083:8	21:203	The vehicle for the present guided tour shall be multitext grammar (MTG), which is a generalization of context-free grammar to the synchronous case (Melamed, 2003).
---------------------------------------------------
P04-1083:9	54:203	E.g., a53a45a54a55 is a vector spanning dimensions 1 through a56 . See Melamed (2003) for definitions of cardinality, d-span, and the operators a57 and a58 . Parser C needs to know all the boundaries of each item, not just the outermost boundaries.
---------------------------------------------------
P04-1083:10	189:203	For example, Melamed (2003) showed how to reduce the computational complexity of a synchronous parser by a7 a23 a0 a51 a24, just by changing the logic.
---------------------------------------------------
E06-1019:11	11:254	Someworkwithintheframework of synchronous grammars (Wu, 1997; Melamed, 2003), while others create a generative story that includes a parse tree provided for one of the sentences (Yamada and Knight, 2001).
---------------------------------------------------
E06-1019:12	245:254	These switches correspond to discontinuous constituents (Melamed, 2003) in general bitext parsing.
---------------------------------------------------
W07-0403:13	70:234	(2) cannot represent all possible permutations of concepts that may occur during translation, because some permutations will require discontinuous constituents (Melamed, 2003).
---------------------------------------------------
D08-1089:14	19:184	Coming from the other direction, such observations about phrase reordering between different languages are precisely thekindsoffactsthatparsingapproachestomachine translation are designed to handle and do successfully handle (Wu, 1997; Melamed, 2003; Chiang, 2005).
---------------------------------------------------
P06-2036:15	8:153	Variations of SCFGs go back to Aho and Ullman (1972)s Syntax-Directed Translation Schemata, but also include the Inversion Transduction Grammars in Wu (1997), which restrict grammar rules to be binary, the synchronous grammars in Chiang (2005), which use only a single nonterminal symbol, and the Multitext Grammars in Melamed (2003), which allow independent rewriting, as well as other tree-based models such as Yamada and Knight (2001) and Galley et al.
---------------------------------------------------
W05-0803:16	13:160	The only requirement will be that a parallel corpus exist for the language under consideration and one or more other languages.2 Induction of grammars from parallel corpora is rarely viewed as a promising task in its own right; in work that has addressed the issue directly (Wu, 1997; Melamed, 2003; Melamed, 2004), the synchronous grammar is mainly viewed as instrumental in the process of improving the translation model in a noisy channel approach to statistical MT.3 In the present paper, we provide an important prerequisite for parallel corpus-based grammar induction work: an efficient algorithm for synchronous parsing of sentence pairs, given a word alignment.
---------------------------------------------------
W05-0803:17	149:160	20While our simulation may be significantly slower than a direct implementation of the algorithm (especially when some of the optimizations discussed in (Melamed, 2003) are taken into account), the fact that it is orders of magnitude slower does in23 Finally, we also ran an experiment in which the continuity condition (condition (iii) in rule (4)) was deactivated, i.e., complete constituents were allowed to be discontinuous in one of the languages.
---------------------------------------------------
W05-0803:18	69:160	(2) X1/X2    Y1:r1/Y2:r2 , [i1, j1, i2, j2], Y1/Y2   , [j1, k1, j2, k2] X1/X2   Y1:r1/Y2:r2  , [i1, k1, i2, k2] (3) X1/X2    Y1:r1/Y2:r2 , [i1, j1, j2, k2], Y1/Y2   , [j1, k1, i2, j2] X1/X2   Y1:r1/Y2:r2  , [i1, k1, i2, k2] Since each inference rule contains six free variables over string positions (i1, j1, k1, i2, j2, k2), we get a parsing complexity of order O(n6) for unlexicalized grammars (where n is the number of words in the longer of the two strings from language L1 and L2) (Wu, 1997; Melamed, 2003).
---------------------------------------------------
W05-0803:19	35:160	As (Melamed, 2003) discusses for instance, in the context of binary branching structures even simple examples like the English/French pair a gift for you from France  un cadeau de France pour vouz [a gift from France for you] lead to discontinuity of a synchronous phrase in one of the two languages.
---------------------------------------------------
W05-0803:20	61:160	Graphically speaking, parsing amounts to identifying rectangular crosslinguistic constituents  by assembling smaller rectangles that will together cover the full string spans in both dimensions (compare (Wu, 1997; Melamed, 2003)).
---------------------------------------------------
W05-0803:21	74:160	As an alternative to the standard rectangular indexing approach 10The assumption here (following (Melamed, 2003)) is that lexicalization is not considered as just affecting the grammar constant, but that in parsing, every terminal symbol has to be considered as the potential head of every phrase of which it is a part.
---------------------------------------------------
H05-1101:22	18:170	Among the several proposals, we mention here the models presented in (Wu, 1997; Wu and Wong, 1998), (Alshawi et al. , 2000), (Yamada and Knight, 2001), (Gildea, 2003) and (Melamed, 2003).
---------------------------------------------------
W05-1507:23	56:128	Melamed (2003) discussed the applicability of the hook trick for parsing bilexical multitext grammars.
---------------------------------------------------
N06-1033:24	11:186	Melamed (2003) discusses binarization of multitext grammars on a theoretical level, showing the importance and dif culty of binarization for ef cient synchronous parsing.
---------------------------------------------------
P06-2122:25	10:174	Lexicalization seems likely to help models predict alignment patterns between languages, and has been proposed by Melamed (2003) and implemented by Alshawi et al.
---------------------------------------------------
P06-2122:26	46:174	However, modeling complete bilingual bilexical dependencies as theorized in Melamed (2003) implies a huge parameter space of O(|V |2|T|2), where |V | and |T| are the vocabulary sizes of the two languages.
---------------------------------------------------
P06-2122:27	79:174	Melamed (2003) discussed the applicability of the so-called hook trick for parsing bilexical multitext grammars.
---------------------------------------------------
P06-2122:28	7:174	Synchronous grammar formalisms that are capable of modeling such complex relationships while maintaining the context-free property in each language have been proposed for many years, (Aho and Ullman, 1972; Wu, 1997; Yamada and Knight, 2001; Melamed, 2003; Chiang, 2005), but have not been scaled to large corpora and long sentences until recently.
---------------------------------------------------
P09-1111:29	13:224	rewriting, up to some bounded degree, and have recently been exploited, in some syntactic variant, in syntax-based machine translation (Chiang, 2005; Melamed, 2003) as well as in the modeling of syntax-semantic interface (Nesson and Shieber, 2006).
---------------------------------------------------
H05-1036:30	317:369	These techniques included unweighted FS morphology, conditional random fields (Lafferty et al. , 2001), synchronous parsers (Wu, 1997; Melamed, 2003), lexicalized parsers (Eisner and Satta, 1999),22 partially supervised training `a la (Pereira and Schabes, 1992),23 and grammar induction (Klein and Manning, 2002).
---------------------------------------------------
P05-1067:31	24:217	Melamed (2003, 2004) formalized the MT problem as synchronous parsing based on multitext grammars.
---------------------------------------------------
P04-1084:32	43:180	GMTG can express their relationships as follows: a0a6a5 S a9a11a1 a5 S a9a13a2a15a3 a0a6a5 NP a7 VPa8a10a9a11a1 a5 NP a7 VPa8a38a9a13a2 (12) a0a6a5 VP a9a11a1 a5 VP a9a13a2a4a3 a0a6a5 V a7 NPa8 a9a11a1 a5 V a7 PPa8 a9a13a2 (13) a0a14a5 NP a9a11a1 a5 PP a9a13a2a4a3 a0a6a5 DT a7 Aa8a39a1 Na40a10a9a11a1 a5 VB a41a42a1 Ra43a10a9a13a2 (14) a0a6a5 NP a9a11a1 a5 NP a9a13a2a4a3 a0a6a5 Tim a9a11a1 a5 Tim a9a13a2 (15) a0a44a5 V a9a11a1 a5 V a9a13a2a15a3 a0a6a5 got a9a11a1 a5 got a9a13a2 (16) a0a44a5 DT a9a11a1 a5 a9a13a2a16a3 a0a6a5 a a9a11a1 a5 a9a13a2 (17) a0a6a5 A a9a11a1 a5 a9a13a2a16a3 a0a6a5 pink a9a11a1 a5 a9a13a2 (18) a0a6a5 N a9a11a1 a5 a9a13a2a16a3 a0a6a5 slip a9a11a1 a5 a9a13a2 (19) a0a44a5 a9a11a1 a5 VB a9a13a2a15a3 a0a6a5 a9a11a1 a5 laid a9a13a2 (20) a0a6a5 a9a11a1 a5 R a9a13a2a15a3 a0a6a5 a9a11a1 a5 off a9a13a2 (21) As described by Melamed (2003), MTG requires production components to be contiguous, except after binarization.
---------------------------------------------------
P04-1084:33	154:180	Thus, GCNF is a more restrictive normal form than those used by Wu (1997) and Melamed (2003).
---------------------------------------------------
P04-1084:34	13:180	More technically, our proposal starts from Multitext Grammar (MTG), a formalism for synchronizing context-free grammars recently proposed by Melamed (2003).
---------------------------------------------------
P04-1084:35	6:180	As shown by Melamed (2003), a plausible model of parallel text must be able to express discontinuous constituents.
---------------------------------------------------
