4 Discriminative Model of Identifying Term Variation Recent work in spelling correction (Ahmed and Kondrak, 2005; Li et al., 2006; Chen et al., 2007) and normalization (Okazaki et al., 2008b) formulates the task in a discriminative framework:    = argmax  gen     ( | ) This model consists of two components: gen(q) generates a list of candidates C(q) for an input query q, which are then ranked by the ranking function P(c|q).


In particular, our work is closely related to research in spelling correction for English web queries (e.g., Cucerzan and Brill, 2004; Ahmad and Kondrak, 2005; Li et al., 2006; Chen et al., 2007).


Recent studies have formalized the task in the discriminative framework (Ahmad and Kondrak, 2005; Li et al., 2006; Chen et al., 2007), t = argmax tgen(s) P(t|s).


(1) Here, the candidate generator gen(s) enumerates candidates of destination (correct) strings, and the scorer P(t|s) denotes the conditional probability of the string t for the given s. The scorer was modeled by a noisy-channel model (Shannon, 1948; Brill and Moore, 2000; Ahmad and Kondrak, 2005) and maximum entropy framework (Berger et al., 1996; Li et al., 2006; Chen et al., 2007).


