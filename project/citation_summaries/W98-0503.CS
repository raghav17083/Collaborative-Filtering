This measure corresponds to the tree-based gap degree measure in (Kuhlmann and Nivre, 2006), which was first introduced in (Holan et al. , 1998) there it is defined as the maximum over gap degrees of all subtrees of a dependency tree (the gap degree of a subtree is the number of contiguous intervals in the gap of the subtree).


Definition 3 For any edge i  j in a dependency tree T we define its gap as follows Gapij ={vV |v(i, j) & v /Subtreeij} . An edge with an empty gap is projective, an edge whose gap is non-empty is non-projective.5 We see that non-projective are those edges i  j for which there is a node v such that together they violate the condition of Harper & Hays; we group all such nodes v into Gapij, the gap of the nonprojective edge i j. The notion of gap is defined differently for subtrees of a dependency tree (Holan et al. , 1998; Bodirsky et al. , 2005).


In the fourth approach, the dependency grammar is converted into a headed context-free grammar (Abney, 1996; Holan et al. , 1998), also the Basic Dependency Grammar of Beletskij (1967) as cited in (Dikovsky and Modina, 2000).


