Algorithm 1 MBR over Sentence Pairs 1: A 2: for eE do 3: Ae0 4: for eprime E do 5: AeAe + P(eprime|f)S(e;eprime) 6: if Ae > A then A,eAe,e 7: return e We can sometimes exit the inner for loop early, whenever Ae can never become larger than A (Ehling et al., 2007).


Further experiments using MBR for Bleu were performed in (Venugopal et al. , 2005; Ehling et al. , 2007).


The MBR hypotheses were generated using the algorithm described in (Ehling et al. , 2007) on a 10000-best list.


There has been prior work in MBR speech recognition and machine translation (Goel and Byrne, 2000; Ehling et al., 2007) which has shown the need for tuning this factor.


