Comparable corpora have been studied extensively in the literature (e.g. ,(Fung, 1995; Rapp, 1995; Tanaka and Iwasaki, 1996; Franz et al. , 1998; Ballesteros and Croft, 1998; Masuichi et al. , 2000; Sadat et al. , 2003)), but transliteration in the context of comparable corpora has not been well addressed.


2 Two-stages Comparable Corpora-based Approach Our proposed approach to bilingual terminology acquisition from comparable corpora (Sadat et al. , 2003; Sadat et al. , 2003) is based on the assumption of similar collocation, i.e., If two words are mutual translations, then their most frequent collocates are likely to be mutual translations as well.


The merging process will keep common pairs of source term and target translation (s,t) which appear in SIMS!T as pairs of (s,t) but also in SIMT!S as pairs of (t,s), to result in combined similarity vectors SIMS$T for each pair (s,t).The product of similarity values of both similarity vectors SIMS!T for pairs (s,t) and SIMT!S for pairs (t,s) will result in similarity values in vectors SIMS$T . Therefore, similarity vectors of the two-stages translation model are expressed as follows: SIMS$T = f(s;t;simS$T (tjs)) j (s;t;simS!T (tjs)) 2 SIMS!T ^ (t;s;simT!S(sjt)) 2 SIMT!S ^ simS$T (tjs) = simS!T (tjs) simT!S(sjt)g 3 Linguistics-based Pruning Combining linguistic and statistical methods is becoming increasingly common in computational linguistics, especially as more corpora become available (Klanvans and Tzoukermann, 1996; Sadat et al. , 2003).


The proposed two-stages approach for the acquisition, disambiguation and selection of bilingual terminology is described as follows: Bilingual terminology acquisition from source language to target language to yield a first translation model, represented by similarity SIMS!T. Bilingual terminology acquisition from target language to source language to yield a second translation model, represented by similarity SIMT!S. Merge the first and second models to yield a two-stages translation model, based on bidirectional comparable corpora and represented by similarity SIMS$T . We follow strategies of previous researches (Dejean et al. , 2002; Fung, 2000; Rapp, 1999) for the first and second translation models and propose a merging strategy for the two-stages translation model (Sadat et al. , 2003).


