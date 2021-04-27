THE PARSERS The three parsers in this study are: a bottomup left-corner parser, a (non-deterministic) LR parser, and an LR-like parser based on an algorithm devised by Schabes (1991). 
6Due to the lack of space, the details of the implementation are not given in this paper but they are given in (Schabes, 1991). 
Several other methods (such as LR(k)-like, SLR(k)like) can also be used for constructing the parsing tables and are described in (Schabes, 1991). 
THE COMPILED-EARLEY PARSER The Compiled-Earley (CE) parser is based on a predictive chart-based CF parsing algorithm devised by Schabes (1991) which is driven by a table compiling out the predictive component of Earley's (1970) parser. 
We have adapted the technique presented in this paper to other grammatical formalism such as treeadjoining grammars (Schabes, 1991). 
