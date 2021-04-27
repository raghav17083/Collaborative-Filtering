Different methods have been proposed to reduce error propagation between pipelined tasks, both in general (Sutton et al., 2004; Daume III and Marcu, 2005; Finkel et al., 2006) and for specific problems such as language modeling and utterance classification (Saraclar and Roark, 2005) and labeling and chunking (Shimizu and Haas, 2006).


A single oracle with 1-best translation is analytically solved without a QP-solver and is represented as the following perceptron-like update (Shimizu and Haas, 2006):  = max   0, min   C, L(e, e; et) parenleftBig si( f t, e)si( f t, e) parenrightBig ||h( f t, e)h( f t, e)||2       Intuitively, the update amount is controlled by the margin and the loss between the correct and incorrect translations and by the closeness of two translations in terms of feature vectors.


MIRA is successfully employed in dependency parsing (McDonald et al. , 2005) or the joint-labeling/chunking task (Shimizu and Haas, 2006).


4.1 Margin Infused Relaxed Algorithm The Margin Infused Relaxed Algorithm (MIRA) (Crammer et al. , 2006) is an online version of the large-margin training algorithm for structured classification (Taskar et al. , 2004) that has been successfully used for dependency parsing (McDonald et al. , 2005) and joint-labeling/chunking (Shimizu and Haas, 2006).


