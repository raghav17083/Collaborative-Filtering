Probabilities for expansions, transitions and reductions in this model can be defined over trees in a training corpus, transformed and mapped to the random variables in an HHMM (Schuler et al., 2008).


c 2009 Association for Computational Linguistics Positive Results for Parsing with a Bounded Stack using a Model-Based Right-Corner Transform William Schuler Dept. of Computer Science and Engineering Minneapolis, MN schuler@cs.umn.edu Abstract Statistical parsing models have recently been proposed that employ a bounded stack in timeseries (left-to-right) recognition, using a rightcorner transform defined over training trees to minimize stack use (Schuler et al., 2008).


1 Introduction Statistical parsing models have recently been proposed that employ a bounded stack in time-series (left-to-right) recognition, in order to directly and tractably incorporate incremental phenomena such as (co-)reference or disfluency into parsing decisions (Schuler et al., 2008; Miller and Schuler, 2008).


These models make use of a right-corner tree transform, based on the left-corner transform described by Johnson (1998), and are supported by corpus results suggesting that most sentences (in English, at least) can be parsed using a very small stack bound of three to four elements (Schuler et al., 2008).


