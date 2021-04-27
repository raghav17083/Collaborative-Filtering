We use instead the Anchored Learning heuristic, introduced in (Goldberg and Elhadad, 2007).


1142 We show that by using a variant of SVM  Anchored SVM Learning (Goldberg and Elhadad, 2007) with a polynomial kernel, one can learn accurate models for English NP-chunking (Marcus and Ramshaw, 1995), base-phrase chunking (CoNLL 2000), and Dutch Named Entity Recognition (CoNLL 2002), on a heavily pruned feature space.


However, in Goldberg and Elhadad (2007), we suggested that the SVM learner is using the rare lexical features for singling out hard cases rather than for learning meaningful generalizations.


On the other hand, the long tail of rare features is the reason the Kernel Expansion methods requires 2This loss of accuracy in comparison to the PKE approach is to be expected, as (Goldberg and Elhadad, 2007) showed that the effect of removing features prior to the learning stage is much more severe than removing them after the learning stage.


In our third model GTppp we also add the distinction between general PPs and possessive PPs following Goldberg and Elhadad (2007).


