Likewise, it is not clear how to learn from consistently misannotated data; studies often only note the presence of errors or eliminate them from evaluation (e.g., Hogan, 2007), and a previous attempt at correction was limited to POS annotation (Dickinson, 2006).


The method is successful at detecting annotation errors in corpora for three different languages, with precisions of 93% for Swedish, 60% for Czech, and 48% for German.1 2.2 Error correction Correcting POS annotation errors can be done by applying a POS tagger and altering the input POS tags (Dickinson, 2006).


5.1 Using ambiguity classes Previous error correction work (Dickinson, 2006) used ambiguity classes for POS annotation, and this is precisely the type of information we need to constrain the label to one which we know is relevant to the current case.


Exploring annotation error correction in this way can provide insights into more general uses of the annotation, just as previous work on correction for POS annotation (Dickinson, 2006) led to a way to improve POS 193 tagging (Dickinson, 2007).


This means that 643 positions do not need to be corrected, setting a baseline of 70.1% (643/917) for error correction.2 Following Dickinson (2006), we train our models on the entire corpus, explicitly including NIL relations (see 1The German experiment uses a more relaxed heuristic; precision is likely higher with the shortest context heuristic.


