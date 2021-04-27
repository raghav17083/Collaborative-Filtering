Several exceptions to this described architecture for SRL can be found in the literature.One approach entails joint labeling of all predicates of the sentence, instead of proceeding one by one.This opens the possibility of exploiting dependencies among the different verbs in the sentence.However, the complexity may grow signicantly, and results so far are inconclusive (Carreras, M`arquez, and Chrupaa 2004; Surdeanu et al. 2007).Other promising approaches draw on dependency parsing rather than traditional phrase structure parsing (Johansson and Nugues 2007), or combine parsing and SRL into a single step of semantic parsing (Musillo and Merlo 2006).


(Wong and Mooney, 2007)) or jointly learning the syntactic structure of the sentence and the propositional argument-structure of its main predicates (Musillo and Merlo, 2006; Merlo and Musillo, 2008).


This inter-connectivity depends on a notion of structural locality (Henderson, 2003; Musillo and Merlo, 2006).2 2Specifically, the conditioning states are based on the In order to extend this model to learn decisions concerning a joint syntactic-semantic representation, the semantic information needs to be highlighted in the model in several ways.


(Musillo and Merlo, 2006) report results of a merging technique where the output of the semantic role annotation produced by the best semantic role labellers in the 2005 CONLL shared task is merged with the output of Charniaks parser.


Other works have integrated argument classification and identification into one step (Collobert and Weston, 2007), while others went further and combined the former two along with parsing into a single model (Musillo 29 and Merlo, 2006).


