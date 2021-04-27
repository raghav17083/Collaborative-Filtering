This paper describes an evaluation of a focusbased approach to pronoun resolution (not anaphora in general), based on aI1 extension of Sidner's algorithm (Sidner, 1981) proposed in (Azzam, 1996), with further refinements from development on real-world texts.


This example is taken from a New York Times article in the MUC-7 training corpus on aircraft crashes: 1 An important limitation of Sidner's algorithm, noted in (Azzam, 1996), is that the focus registers are only updated after each sentence.


4 Implementing Focus-Based Pronoun Resolution in LaSIE Our implementation makes use of the algorithm proposed in (Azzam, 1996), where elementary events (EEs, effectively simple clauses) are used as basic processing units, rather than sentences.


The approach * This work was carried out in the context of the EU AVENTINUS project (Thumair, 1996), which aims to develop a multilingual IE system for drug enforcement, and including a language-independent coreference mechanism (Azzam et al., 1998).


