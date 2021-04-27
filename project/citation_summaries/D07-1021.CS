Either pruning (Stolcke, 1998; Church et al., 2007) or lossy randomizing approaches (Talbot and Brants, 2008) may result in a compact representation for the application run-time.


(Emami et al., 2007), (Brants et al., 2007), (Church et al., 2007).


To scale LMs to larger corpora with higher-order dependencies, researchers Work completed while this author was at Google Inc. have considered alternative parameterizations such as class-based models (Brown et al., 1992), model reduction techniques such as entropy-based pruning (Stolcke, 1998), novel represention schemes such as suffix arrays (Emami et al., 2007), Golomb Coding (Church et al., 2007) and distributed language models that scale more readily (Brants et al., 2007).


Table 2 shows the total space and number of bytes required per n-gram to encode the model under different schemes: LDC gzipd is the size of the files as delivered by LDC; Trie uses a compact trie representation (e.g., (Clarkson et al., 1997; Church et al., 2007)) with 3 byte word ids, 1 byte values, and 3 byte indices; Block encoding is the encoding used in (Brants et al., 2007); and randomized uses our novel randomized scheme with 12 error bits.


Note that unlike the constructions in (Talbot and Osborne, 2007b) and (Church et al., 2007) no errors are possible for ngrams stored in the model.


