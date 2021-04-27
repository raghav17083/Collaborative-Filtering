We use a state-of-the-art phrase-based translation system as described in (Zens and Ney, 2004; Zens et al. , 2005). 
Above the phrase level, these models typically have a simple distortion model that reorders phrases independently of their content (Och and Ney, 2004; Koehn et al. , 2003), or not at all (Zens and Ney, 2004; Kumar et al. , 2005). 
4.1 The Monotone Search Algorithm Zens and Ney (2004) propose a linear-time decoding algorithm for phrase-based machine translation. 
This is the traditional approach for glass-box smoothing (Koehn et al. , 2003; Zens and Ney, 2004). 
We use two different estimates for the conditional probabilities p(?t|?s) and p(?s|?t): relative frequencies and ?lexical??probabilities as described in (Zens and Ney, 2004). 
