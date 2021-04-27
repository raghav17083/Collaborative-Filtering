1c, (2) by compilation of weighted rewrite rules (Mohri and Sproat, 1996), (3) by compilation of decision trees (Sproat and Riley, 1996), (4) as a relation that performs contextual left-to-right replacement of input substrings by a smaller conditional relation (Gerdemann and van Noord, 1999),5 (5) by conditionalization of a joint relation as discussed below.


Gerdemann and van Noord (1999) implement leftmost longest-match replacement in FSA as the operator replace(Target, LeftContext,RightContext), where Target is a transducer defining the actual replacement, and LeftContext and RightContext are regular expressions defining the leftand rightcontext of the rule, respectively.


The FSA Utilities toolbox (van Noord 1997, 1999; Gerdemann and van Noord 1999; van Noord and Gerdemann 1999) is a collection of tools to manipulate regular expressions, finite-state automata, and finite-state transducers.


