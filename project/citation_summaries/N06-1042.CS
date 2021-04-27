For the word emeli for instance, our morphological analyzer generates the following with lexical morphemes bracketed with (): (em)em+Verb+Pos(+yAlH)?DB+Adverb+Since since (someone) sucked (something) (emel)emel+Noun+A3sg(+sH)+P3sg+Nom his/her ambition (emel)emel+Noun+A3sg+Pnon(+yH)+Acc ambition (as object of a transitive verb) These analyses are then disambiguated with a statistical disambiguator (Yuret and Ture, 2006) which operates on the morphological features.2 Finally, the morphological features are removed from each parse leaving the lexical morphemes.


In this case we activate the part-of-speech tagger of Yuret and Ture (2006) which uses some rules automatically derived from a training corpus.


246 2 The Learning Algorithm The Greedy Prepend Algorithm (Yuret and Ture, 2006) was used to build decision lists to identify dependency relations.


GPA was picked for this problem because it has proven to be fairly ef cient and robust in the presence of irrelevant or redundant attributes in previous work such as morphological disambiguation in Turkish (Yuret and Ture, 2006) and protein secondary structure prediction (Kurt, 2005).


