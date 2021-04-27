Notice that q here does not approximate the entire translation process, but only 2May and Knight (2006) have successfully used treeautomaton determinization to exactly marginalize out some of the nuisance variables, obtaining a distribution over parsed translations.


Modifying (2) to sum over only these derivations is called crunching by May and Knight (2006): y = argmax yT(x) pcrunch(y|x) (7) = argmax yT(x) summationdisplay dD(x,y)ND(x) p(y,d|x) 3 Variational Approximate Decoding The Viterbi and crunching methods above approximate the intractable decoding of (2) by ignoring most of the derivations.


Furthermore, there are algorithms for determinizing weighted tree automata (Borchardt and Vogler, 2003; May and Knight, 2006), which could be applied as preprocessing steps for wRTGs.


May and Knight (2006) extract nbest lists containing unique translations rather than unique derivations, while Kumar and Byrne (2004) use the Minimum Bayes Risk decision rule to select the lowest risk (highest BLEU score) translation rather than derivation from an n-best list.


To alleviate this problem, determinization techniques have been proposed by Mohri and Riley (2002) for finite-state automata and extended to tree automata by May and Knight (2006).


