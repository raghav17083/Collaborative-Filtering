(Gimpel and Smith, 2008; Max et al., 2008; Haque et al., 2009)).


Furthermore, based on the hypothesis that a paraphrase for an unmodified envelope should preserve the syntactic dependencies between the paraphrased fragment and its envelope (inter-fragment dependencies), we optionaly add a hard reranking step where we filter the N-best list of hypothe7We will integrate richer syntactic context as in (Gimpel and Smith, 2008; Max et al., 2008) in our short-term future work, as we expect it to be particularly useful for our paraphrasing task.


A typical contextual score corresponds to p(e|f,C(f)), where C(f) is some contextual information about the source phrase f. An external disambiguation system can be used to provideoneglobalcontextscore(Stroppaetal., 2007; Carpuat and Wu, 2007; Max et al., 2008)); alternatively, several scores based on single features can be estimated using relative frequencies (Gimpel and Smith, 2008): p(e|f,C(f)) = count(e,f,C(f))summationtext eprime count(eprime,f,C(f)) For these experiments, we followed the latter approach, restricting ourselves to features representing the local context up to a fixed distance d (using the values 1 and 2 in our experiments) from the source phrase fendstart:  lexical context features:  left context: p(e|f,fstart1startd)  right context: p(e|f,fend+dend+1)  shallow syntactic features (denoting tF1 the sequence of POS tags for the source sentence):  left context: p(e|f,tstart1startd)  right context: p(e|f,tend+dend+1) As in (Gimpel and Smith, 2008), we filtered out all translations for which p(e|f) < 0.0002.


In fact, attempts at using source contexts in phrase-based SMT have to date failed to show important gains on standard evaluation test sets (Carpuat and Wu, 2007; Stroppa et al., 2007; Gimpel and Smith, 2008; Max et al., 2008).


In addition, researchers have recently added syntactic features to phrase-based and hierarchical phrase-based models (Gimpel and Smith, 2008; Haque et al., 2009; Chiang et al., 2008).


