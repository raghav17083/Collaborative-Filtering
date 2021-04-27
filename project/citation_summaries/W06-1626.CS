(Schwenk and Koehn, 2008; Zhang et al., 2006) used higher language models at time of re-ranking rather than integrating directly into the decoder to avoid the overhead of keeping LMs in the main memory since disk lookups are simply too slow.


Zhang et al.(2006) and Mauser et al.(2006) use adapted language model for SMT re-ranking.


Recently a two-pass approach has been proposed (Zhang et al. , 2006), wherein a lowerorder n-gram is used in a hypothesis-generation phase, then later the K-best of these hypotheses are re-scored using a large-scale distributed language model.


The underlying architecture is similar to (Zhang et al. , 2006).


It is therefore desirable to have dedicated servers to load parts of the LM3  an idea that has been exploited by (Zhang et al., 2006; Emami et al., 2007; Brants et al., 2007).


