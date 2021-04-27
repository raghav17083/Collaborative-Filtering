Previous approaches to the problem (Collins, 1997; Johnson, 2002; Dienes and Dubey, 2003a,b; Higgins, 2003) have all been learning-based; the primary difference between the present algorithm and earlier ones is that it is not learned, but explicitly incorporates principles of GovernmentBinding theory (Chomsky, 1981), since that theory underlies the annotation.


1 Empty categories however seem different, in that, for the most part, their location and existence is determined, not by observable data, but by explicitly constructed linguistic principles, which 1 Both Collins (1997: 19) and Higgins (2003: 100) are explicit about this predisposition.


2 Higgins (2003) considers only the detection and resolution of WH-traces, and only evaluates the results given perfect input.


Three strategies have been proposed: (i) post-processing parser output with pattern matchers (Johnson, 2002), linguistic principles (Campbell, 2004) or machine learning methods (Higgins, 2003; Levy and Manning, 2004; Gabbard et al. , 2006) to recover empty nodes and identify their antecedents;1 (ii) integrating non-local dependency recovery into the parser by enriching a simple PCFG model with GPSG-style gap features (Collins, 1999; Schmid, 2006); (iii) pre-processing the input sentence with a finite-state trace tagger which detects empty nodes before parsing, and identify the antecedents on the parser output with the gap information (Dienes and Dubey, 2003a; Dienes and Dubey, 2003b).


