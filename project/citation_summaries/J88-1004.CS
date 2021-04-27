183 2.7 Finite Domains For domains involving only a finite set of atoms as possible values, it is possible to provide a Prolog term representation (due to Colmerauer, and described by Mellish (Mellish, 1988)) to encode any subset of the possible values in one term. 
An example of a systemic choice network in the notation of Mellish (1988), is Figure 4. 
Mellish (1988) reports an application in which this route was taken with satisfactory results. 
3 From ProFIT terms to Prolog terms 3.1 Compilation of Sorted Feature Terms The compilation of sorted feature terms into a Prolog term representation is based on the following principles, which are explained in more detail in (Mellish, 1988; Mellish, 1992; Schgter, 1993; Erbach, 1994). 
184 $sign(yar, $phrasal(Phrasesort,Clausesort,Dtrs), Phon, Synsem, Qstore, Retrieved) 3.2 Compilation of Finite Domains The compilation of finite domains into Prolog terms is performed by the "brute-force" method described in (Mellish, 1988). 
