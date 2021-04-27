J07-1003:1	39:616	Nowadays, most state-of-the-art SMT systems are based on bilingual phrases (Och, Tillmann, and Ney 1999; Koehn, Och, and Marcu 2003; Tillmann 2003; Bertoldi et al. 2004; Vogel et al. 2004; Zens and Ney 2004; Chiang 2005).
---------------------------------------------------
I08-1064:2	144:193	Although bi-alignments are known to exhibit high precision (Koehn et al., 2003), in the face of sparse annotations we use unidirectional alignments as a fallback, as has been proposed in the context of phrase-based machine translation (Koehn et al., 2003; Tillmann, 2003).
---------------------------------------------------
H05-1023:3	191:217	The block extraction procedure is based on the procedures in (Tillmann, 2003).
---------------------------------------------------
H05-1023:4	202:217	A monotone decoder similar to (Tillmann and Ney, 2003) with a trigram language model3 is set up for translations.
---------------------------------------------------
H05-1023:5	157:217	We collect bidirectional (bi) refined word alignment by growing the intersection of Chinese-to-English (CE) alignments and English-to-Chinese (EC) alignments with the neighboring unaligned word pairs which appear in the union similar to the final-and approaches (Koehn, 2003; Och and Ney, 2003; Tillmann, 2003).
---------------------------------------------------
W05-0712:6	91:175	4.2 Phrase Based Named Entity Translation For phrase-based NE translation, we used an approach similar to that presented by Tillman (Tillmann, 2003) for block generation with modifications suitable for NE phrase extraction.
---------------------------------------------------
W05-0712:7	121:175	We used a dynamic programming beam search decoder similar to the decoder described by Tillman (Tillmann, 2003).
---------------------------------------------------
W05-0712:8	97:175	The reader is referred to (Tillmann, 2003) for detailed description of the algorithm.
---------------------------------------------------
D08-1060:9	139:222	Then, phrase-pairs that overlap with our development and test set were extracted from the word alignments (from both hand alignments and automatically aligned GALE corpora) based on the projection principle (Tillmann, 2003).
---------------------------------------------------
W05-0825:10	9:75	A widely practiced approach explained in details in (Koehn, 2004), (Och and Ney, 2003) and (Tillmann, 2003) is to get word alignments from two directions: source to target and target to source; the intersection or union operation is applied to get re ned word alignment with pre-designed heuristics xing the unaligned words.
---------------------------------------------------
J04-4002:11	243:482	For single-word-based search in MT, a similar algorithm has been described in Tillmann and Ney (2003).
---------------------------------------------------
J04-4002:12	37:482	Looking at the results of the recent machine translation evaluations, this approach seems currently to give the best results, and an increasing number of researchers are working on different methods for learning phrase translation lexica for machine translation purposes (Marcu and Wong 2002; Venugopal, Vogel, and Waibel 2003; Tillmann 2003; Koehn, Och, and Marcu 2003).
---------------------------------------------------
H05-1096:13	29:156	Nowadays, most of the state-of-the-art SMT systems are based on bilingual phrases (Bertoldi et al. , 2004; Koehn et al. , 2003; Och and Ney, 2004; Tillmann, 2003; Vogel et al. , 2004; Zens and Ney, 2004).
---------------------------------------------------
P05-1074:14	46:147	Koehn (2004), Tillmann (2003), and Vogel et al.
---------------------------------------------------
W06-3602:15	63:191	The decoder in Table 1 fills a115a152a131 a113 state sets a94a118a107a153a100a101a111a154a15 a103a3a116 a23 a30a97a30a50a30 a23a97a115 a106 . All the coverage vectorsa79 for states in the set a94a86a107 cover the same number of source positions a111 . When a state set a94a118a107 is processed, the decoder has finished processing all states in the sets a94 a125 where a155a157a156a158a111 . Before expanding a state set, the decoder prunes a state set based on its coverage vector and the path costs only: two different pruning strategies are used that have been introduced in (Tillmann and Ney, 2003): 1) coverage pruning prunes states that share the same coverage vector a79, 2) cardinality pruning prunes states according to the cardinality a19 a4a159a79a150a11 of covered positions: all states in the beam are compared with each other.
---------------------------------------------------
W06-3602:16	12:191	is carried out in one pass over some input along a specific direction: in speech recognition the search is timesynchronous, the single-word based search algorithm in (Tillmann et al. , 1997) is (source) position-synchronous or left-to-right, the search algorithm in (Niessen et al. , 1998) is (target) position-synchronous or bottom-to-top, and the search algorithm in (Tillmann and Ney, 2003) is so-called cardinality-synchronous.
---------------------------------------------------
W06-3602:17	109:191	Without proof, for the search algorithm in Section 2.1 we observe that the number of states is finite and that all the states are actually reachable from the start state a105a87a98 . This way for the single-word based search in (Tillmann and Ney, 2003), a complexity of a174a101a4a60a175a18a172 a28 a175a177 a30 a115a147a178 a30 a179 a96 a11 is shown, where a175a18a172 a28 a175 is the size of the target vocabulary and a115 is the length of the input sentence.
---------------------------------------------------
W06-3602:18	166:191	Beam-search results are presented in terms of two pruning thresholds: the coverage pruning threshold a220 a222 and the cardinality pruning threshold a220 a95 (Tillmann and Ney, 2003).
---------------------------------------------------
W06-3602:19	4:191	The paper contains the following original contributions: 1) the DP-based decoding algorithm in (Tillmann and Ney, 2003) is extended in a formal way to handle phrases and a novel pruning strategy with increased translation speed is presented 2) a novel alignment algorithm is presented that computes a phrase alignment efficiently in the case that it is consistent with an underlying word alignment.
---------------------------------------------------
W06-3602:20	18:191	In this paper, the DP-based search algorithm in (Tillmann and Ney, 2003) is extended in a formal way to handle phrase-based translation.
---------------------------------------------------
W06-3602:21	121:191	To restrict the block selection based on word aligned training data, interval projection functions are defined as follows 4: a140 is a source interval and a148 is an target inter3(Tillmann, 2003) reports an intersection coverage of about a186a168a187 % for Arabic-English parallel data, and a coverage of a188a97a189 % for Chinese-English data.
---------------------------------------------------
W06-3602:22	7:191	Dynamic Programming based search algorithms are being used in speech recognition (Jelinek, 1998; Ney et al. , 1992) as well as in statistical machine translation (Tillmann et al. , 1997; Niessen et al. , 1998; Tillmann and Ney, 2003).
---------------------------------------------------
W06-3602:23	159:191	For all re-ordering experiments, the notion of skips is used (Tillmann and Ney, 2003) to restrict the phrase re-ordering: the number of skips restricts the number of holes in the coverage vector for a left-to-right traversal of the input sentence.
---------------------------------------------------
N04-4015:24	65:115	5 4.2 Phrase Translation Model Impact of Arabic morphological analysis on a phrase translation model with monotone decoding (Tillmann 2003), is shown in Table 3.
---------------------------------------------------
P05-1032:25	20:147	Tillmann (2003) further examines the relationship between maximum phrase length, size of the translation table, and accuracy of translation when inducing block-based phrases from word-level alignments.
---------------------------------------------------
