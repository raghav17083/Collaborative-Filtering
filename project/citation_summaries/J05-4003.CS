Others extract parallel sentences from comparable or non-parallel corpora (Munteanu and Marcu 2005, 2006). 
We are aiming at finding the T with the highest probability p(c = 1|S, T), but we cannot compute that probability for all sentence 1In addition, the sentence length filter in (Munteanu and Marcu, 2005) is used: the length ratio max(J,I)/min(J,I) of source and target sentence has to be smaller than 2. 
Currently, we are working on a feature-rich approach (Munteanu and Marcu, 2005) to improve the sentence-pair selection accuracy. 
3.2 Extraction Experiments On each of our comparable corpora, and using each of our initial parallel corpora, we apply both the fragment extraction and the sentence extraction method of Munteanu and Marcu (2005). 
 Sentence-level filter: The word-overlap filter in (Munteanu and Marcu, 2005) has been implemented: for a sentence pair (S,T) to be considered parallel the ratio of the lengths of the two sentences has to be smaller than two. 
