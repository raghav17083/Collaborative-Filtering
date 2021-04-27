The second and more closely related class deals with extracting specific facts such as birthplace, occupation, etc. For this task, the primary theme of work in the literature has been to treat the task as a general semantic-class learning problem where one starts with a few seeds of the semantic relationship of interest and learns contextual patterns such as <NAME> was born in <Birthplace> or <NAME> (born <Birthdate>) (Hearst, 1992; Riloff, 1996; Thelen and Riloff, 2002; Agichtein and Gravano, 2000; Ravichandran and Hovy, 2002; Mann and Yarowsky, 2003; Jijkoun et al., 2004; Mann and Yarowsky, 2005; Alfonseca et al., 2006; Pasca et al., 2006).


Such templatic patterns can be learned using seed examples of the attribute in question and, there has been a plethora of work in the seed-based bootstrapping literature which addresses this problem (Ravichandran and Hovy, 2002; Thelen and Riloff, 2002; Mann and Yarowsky, 2005; Alfonseca et al., 2006; Pasca et al., 2006) Thus for our baseline we implemented a standard Ravichandran and Hovy (2002) pattern learning model using 100 seed2 examples from an online biographic database called NNDB (http://www.nndb.com) for each of the biographic attributes: Birthdate, Birthplace, Deathdate, Gender, Nationality, Occupation and Religion.


The approach followed for the generalisation is the one described by (Alfonseca et al. , 2006a; Ruiz-Casado et al. , in press), which has a few modifications with respect to Ravichandran and Hovy (2002)s, such as the use of the wildcard * to represent any sequence of words, and the addition of part-of-speech and Named Entity labels to the patterns.


We assume that the system has used the seed list to extract and generalise a set of patterns for each of the relations using training corpora (Ravichandran and Hovy, 2002; Alfonseca et al. , 2006a).


The training corpora are processed with a partof-speech tagger and a module for Named Entity Recognition and Classification (NERC) that annotates people, organisations, locations, dates, relative temporal expressions and numbers (Alfonseca et al. , 2006b), so this information can be included in the patterns.


