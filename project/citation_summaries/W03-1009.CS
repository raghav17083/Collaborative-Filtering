The third baseline (Sporleder) is a reimplementation of Sporleder & Lapatas 2006 algorithm with the following features: Word and Sentence Distances from the current sentence to the previous paragraph break; Sentence Length and Relative Position (relPos) of the sentence in a text; Quotes encodes whether this and the previous sentences contain a quotation, and whether the quotation is continued in the current sentence or not; Final Punctuation of the previous sentence; Words  the first (word1), the first two (word2), the first three and all words from the sentence; Parsed has positive value in case the sentence is parsed, negative otherwise; Number of S, VP, NP and PP nodes in the sentence; Signature is the sequence of PoS tags with and without punctuation; 269 Children of Top-Level Nodes are two features representing the sequence of syntactic labels of the children of the root of the parse tree and the children of the highest S-node; Branching Factor features express the average number of children of S, VP, NP and PP nodes in the parse; Tree Depth is the average length of the path from the root to the leaves; Per-word Entropy is a feature based on Genzel & Charniaks (2003) observation that paragraph-initial sentences have lower entropy than non-initial ones; Sentence Probability according to a language model computed from the training data; Character-level n-gram models are built using the CMU toolkit (Clarkson & Rosenfeld, 1997).


644  Shannons Entropy of x(bj) computed by using a language model as described in Genzel & Charniak (2003).


 Shannons Entropy of the Paragraph: The motivation for including features encoding the entropyofthesentencesistheobservationthatthe entropy of paragraph initial sentences is lower than the others (Genzel & Charniak, 2003).


Similarly, a number of other effects in both comprehension and production have been modeled as resulting from rational strategies of languages users that take into account all the probabilistic information present in the linguistic signal (Genzel and Charniak, 2002; Genzel and Charniak, 2003; Keller, 2004; Levy and Jaeger, 2007).


