We compare it with the other two schemes in two popular instantiations of MT systems based on SCFGs: the string-based systems (Chiang, 2005; Galley et al. , 2006) where the input is a string to be parsed using the source-side of the SCFG; and the treebased systems (Liu et al. , 2006; Huang et al. , 2006) where the input is a parse tree and is recursively converted into a target string using the SCFG as a tree-transducer.


In the worstcase, a binarized grammar with a source arity of s will require at most (2s+1) free indices in a deduction, because otherwise if one rule needs (2s + 2) 36 indices, then there are s+1 spans, which contradicts the definition of arity (Huang et al. , 2005).4 These deductive systems represent the search space of decoding without a language model.


In this paper, we explore use of the hook trick (Eisner and Satta, 1999; Huang et al. , 2005) to reduce the asymptotic complexity of decoding, and the use of heuristics to guide the search.


We also propose another trick (hook) for further speeding up the decoding with integrated n-gram models (Huang et al. , 2005).


In the case of m-gram integrated decoding, we have to maintain 2(m1) boundary words for each child nonterminal, which leads to a prohibitive overall complexity of O(|w|3+2n(m1)), which is exponential in rule size (Huang et al. , 2005).


