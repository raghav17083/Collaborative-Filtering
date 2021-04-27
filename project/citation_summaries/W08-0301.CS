If one wants to select the minimum-Bayes-risk translation from a lattice (or shared forest) instead of an n-best list (Tromble et al., 2008), or to select an oracle translation from a lattice (Tillmann and Zhang, 2006; Dreyer et al., 2007; Leusch et al., 2008), or to perform discriminative training on all the examples containedinalattice(Taskaretal.,2004),onewould need a metric that can be calculated on the edges of the lattice.


1 Introduction B (Papineni et al., 2002) was one of the first automatic evaluation metrics for machine translation (MT), and despite being challenged by a number of alternative metrics (Melamed et al., 2003; Banerjee and Lavie, 2005; Snover et al., 2006; Chan and Ng, 2008), it remains the standard in the statistical MTliterature.Callison-Burchetal.(2006)havesubjected B to a searching criticism, with two realworld case studies of significant failures of correlation between B and human adequacy/fluency judgments.Bothcasesinvolvecomparisonsbetween statistical MT systems and other translation methods (human post-editing and a rule-based MT system), and they recommend that the use of B be restrictedtocomparisonsbetweenrelatedsystemsor different versions of the same systems.


First, some function words should truly be deleted: for example, the Chinese particle/de and Chinese measure words often have no counterpart in English (Li et al., 2008).


