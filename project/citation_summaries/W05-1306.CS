3 Our corpus There are very few biomedical corpora annotated with anaphora information, and all of them are built from paper abstracts (Cohen et al., 2005), instead of full papers.


(See (Cohen et al. , 2005) for an overview of currently available biomedical corpora).


We consider that the lexicalfeatureshm,hmm, andmmarenotdependent on distance d or dm, nor on sr, gp or num, so: P(hm,hmm,mm|C,fA,fa,d,dm,sr,bm,gp,num)  P(hm,hmm,mm|C,fA,fa,bm) We model num as independent from d, dm, sr, bm, and gp, so: P(num|C,fA,fa,d,dm,sr,bm,gp)  P(num|C,fA,fa) We also assume the semantic features bm, and gp as independent from all features but C: P(bm,gp|C,fA,fa,d,dm,sr)  P(bm,gp|C) We also assume sr to be independent of fA and fa: P(sr|C,fA,fa,d,dm)  P(sr|C,d,dm) The final equation then becomes: P(C|fA,fa,hm,hmm,mm,num,sr,bm,gp,d,dm) = P(C) P(fA|C) P(fa|C,fA) P(d,dm|C,fA,fa) P(sr|C,d,dm) P(bm,gp|C) P(num|C,fA,fa) P(hm,hmm,mm|C,fA,fa,bm) P(fA) P(fa|fA) P(d,dm|fA,fa) P(sr|d,dm) P(bm,gp) P(num|fA,fa) P(hm,hmm,mm|fA,fa,bm) (3) 4 Training There are very few biomedical corpora annotated with anaphora information, and all of them are built from paper abstracts (Cohen et al., 2005), instead of full papers.


5 Experimental Setup 5.1 Corpora GENIA (V3.02), a large publicly available biomedical corpus annotated with biomedical NEs, is widely used in the text mining community (Cohen et al. , 2005).


Advances in the area of gene and protein named entity recognition (NER) have been accelerated by freely available tagged corpora (Kim et al. , 2003, Cohen et al. , 2005, Smith et al. , 2005, Tanabe et al. , 2005).


