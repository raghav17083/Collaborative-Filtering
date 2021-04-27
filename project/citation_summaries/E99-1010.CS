This very simple reordering model is widely used, for instance in (Och et al. , 1999; Koehn, 2004; Zens et al. , 2005). 
The classes used in F and E are automatically trained bilingual classes using the method described in (Och, 1999) and constitute a partition of the vocabulary of source and target language. 
The word classes are computed automatically using another statistical training procedure (Och, 1999) which often produces word classes including words with the same semantic meaning in the same class. 
Implementations are also freely available (Al-Onaizan et al. , 1999; Och and Ney, 2003). 
This feature consists of a 5-gram model of words classes, which is trained from the target side of the bilingual corpus using the statistical classes from (Och, 1999). 
