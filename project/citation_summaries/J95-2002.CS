5 Computing priors and posteriors For SCFGs, a probabilistic Earley parser (Earley, 1970; Stolcke, 1995) provides the basic quantities we need to compute the prior (2) and posterior (3) for each category X. Following Stolcke, we use capital Latin characters to denote non-terminal categories and use lowercase Greek characters to denote (possibly null) sequences of terminals and non-terminals. 
Binarization of rules (Earley, 1970) is necessary to obtain cubic parsing time, and closure of unary chains is required for finding total probability mass (rather than just best parses) (Stolcke, 1995). 
There are efficient algorithms in the literature (Jelinek and Lafferty 1991; Stolcke 1995) for calculating exact string prefix probabilities given a PCFG. 
This naturally suggests a minimum description length (MDL) or Bayesian approach (Stolcke, 1994; Chen, 1995). 
For example, Figure 3 presents an alternative form of weighted top-down parsing using pairs of values, following Stolcke (1995). 
