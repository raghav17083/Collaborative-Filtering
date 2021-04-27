1c, (2) by compilation of weighted rewrite rules (Mohri and Sproat, 1996), (3) by compilation of decision trees (Sproat and Riley, 1996), (4) as a relation that performs contextual left-to-right replacement of input substrings by a smaller conditional relation (Gerdemann and van Noord, 1999),5 (5) by conditionalization of a joint relation as discussed below.


In particular, decision trees can be converted into implicational rules that an expert could inspect and can in principle be compiled back into finite-state machines (Sproat and Riley, 1996), although that would re-introduce the original efficiency problems.


Although these approaches have been more effective than HMMs, there have not been many attempts to represent these models as a WFST, with the exception of the work on compiling decision trees (Sproat and Riley, 1996).


a127 a12a128a90 a42 a126 a42 (7) We note that the due to the difference in the nature of the learning algorithm, compiling decision trees results in a composition of WFSTs representing the rules on the path from the root to a leaf node (Sproat and Riley, 1996), while compiling boosted rules results in a union of WFSTs, which is expected to result in smaller transducers.


