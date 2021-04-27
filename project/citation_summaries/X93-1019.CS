The lightweight procedures in IdentiFinder are SGML recognition, hidde n Markov models, finite state pattern recognition, and SGML output . By heavyweight processing, we mean procedures that depend on global evidence and involve deeper understanding . The SPATTER full parser of English and the new semantic inference procedure are examples . 55 KEY SYSTEM FEATURE S Two key design features of PLUM are : statistical language modeling with the associated learning algorithms, an d partial understanding . The first key feature is the use of statistical modeling to guide processing . For the version of PLUM used in MUC-6, part of speech information was determined by using well-known Markov modelin g techniques embodied in BBN's part-of-speech tagger POST (Weischedel, et al ., 1993).


The system pieces together a model of the whole from the parts of the text it can understand . PLUM PROCESSING STAGE S The PLUM architecture is presented in Figures 1 and 2 . Ovals represent declarative knowledge bases; rectangles represent processing modules . A more detailed description of the system components, their individual outputs, an d their knowledge bases is presented in Ayuso et al ., 1993.


al., 1991, 1993),  learning word and word group associations (Matsukawa, 1993),  an example-based correction technique for segmentation and part-of-speech labeling for Japanes e (Matsukawa, Miller, and Weischedel, 1993), an d  induction of a statistical grammar and parser from training data (Magerman, 1995).


TECHNICAL APPROACH Our approach is a synthesis of statistical and linguistic models of language, embodied for MUC-6 in the PLU M system (Probabilistic Language Understanding Model) . We began this research in 1991 and applied it that year i n MUC-3 . Since that time, we have successfully applied probabilistic models to the following :  part of speech tagging, using Hidden Markov Models (Weischedel, et al ., 1993),  judgments of relevance of text (at the paragraph level) via a log odds model (Ayuso, et al ., 1992),  learning semantic information (selection restrictions, or case frames) from supervised training (Weischedel, et.


