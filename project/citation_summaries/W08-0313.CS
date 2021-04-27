Many strategies have been proposed to integrate morphology information in SMT, including factored translation models (Koehn and Hoang, 2007), adding a translation dictionary containing inflected forms to the training data (Schwenk et al., 2008), entirely replacing surface forms by representations built on lemmas and POS tags (Popovic and Ney, 2004), morphemes learned in an unsupervised manner (Virpojia et al., 2007), and using Porter stems and even 4-letter prefixes for word alignment (Watanabe et al., 2006).


Unlike with factored models (Koehn and Hoang, 2007) or additional translation lexicons (Schwenk et al., 2008), we do not generate the surface form back from the lemma translation, which means that tense, gender and number information are 151 news-dev2009a representation OOV % METEOR BLEU NIST baseline surface form only 2.24 49.05 20.45 6.135 decoding lemma backoff 2.13 49.12 20.44 6.143 word alignment lemma+POS for all 2.24 48.87 20.36 6.145 lemma+POS for adj 2.25 48.94 20.46 6.131 lemma+POS for verbs 2.21 49.05 20.47 6.137 decoding + alignment backoff + all 2.10 48.97 20.36 6.147 backoff + adj 2.12 49.05 20.48 6.140 backoff + verbs 2.08 49.15 20.50 6.148 news-dev2009b representation OOV % METEOR BLEU NIST baseline surface form only 2.52 49.60 21.10 6.211 decoding lemma backoff 2.43 49.66 21.02 6.210 word alignment lemma+POS for all 2.53 49.56 21.03 6.199 lemma+POS for adj 2.52 49.74 21.00 6.213 lemma+POS for verbs 2.47 49.73 21.10 6.217 decoding+alignment backoff + all 2.44 49.59 20.92 6.194 backoff + adj 2.43 49.80 21.03 6.217 backoff + verbs 2.39 49.80 21.03 6.217 Table 2: Evaluation of the decoding backoff strategy, the modified word alignment strategy and their combination Input Meme sil demissionnait, la situation ne changerait pas.


The basic architecture of the system is identical to the one used in the 2008 WMT evaluation (Schwenk et al., 2008), but we did not use two pass decoding and n-best list rescoring with a continuous space language model.


The main differences to the previous years system (Schwenk et al., 2008) are as follows: better usage of SYSTRANs bilingual dictionary in the statistical system, less bilingual training data, additional language model training data (news-train08 as distributed by the organizers), usage of comparable corpora to improve the translation model, and development of a statistical post-editing system (SPE).


In comparison to our previous work (Schwenk et al., 2008), we also included all verbs in the French subjonctif and passe simple tense.


