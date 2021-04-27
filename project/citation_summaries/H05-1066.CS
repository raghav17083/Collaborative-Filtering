For training our model, use used an online large margin algorithm (McDonald et al. , 2005). 
That work extends the maximum spanning tree dependency parsing framework (McDonald et al. , 2005a; McDonald et al. , 2005b) to incorporate features over multiple edges in the dependency graph. 
1.1 RelatedWork There has been extensive work on data-driven dependency parsingfor both projective parsing(Eisner, 1996; Paskin, 2001; Yamada and Matsumoto, 2003; Nivre and Scholz, 2004; McDonaldet al., 2005a) and non-projective parsing systems (Nivre and Nilsson,2005;Hall and Novak, 2005;McDonald et al., 2005b). 
This modification is essential inordertomakeourparserrunintrueO(n2)time,asopposed to (McDonald et al., 2005b). 
Thatisnotthecase: dependencyaccuracyfornonprojective parsing is 90.2% for English (McDonald et al., 2005b), only 0.7% lower than a projective parser (McDonald et al., 2005a) that uses the same set of features and learning algorithm. 
