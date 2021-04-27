D09-1043:1	163:172	Also related to the present work is discriminative training in syntax-based MT (Turian et al., 2007; Watanabe et al., 2007; Blunsom et al., 2008; Chiang et al., 2009).
---------------------------------------------------
P09-1067:2	60:225	4Note that the marginalization for a particular y would be tractable; it is used at training time in certain training objective functions, e.g., maximizing the conditional likelihood of a reference translation (Blunsom et al., 2008).
---------------------------------------------------
D09-1107:3	54:233	The version presented in (Blunsom et al., 2008) scales to more than a hundred thousand short training sentences, but does not integrate a language model and thus has performance that improves upon Hiero without a language model only.
---------------------------------------------------
D09-1107:4	59:233	In practice, this problem can be circumvented by discarding the training sentence pairs with unreachable reference translations, but this may mean a significant reduction in the amount of training data (24% in (Blunsom et al., 2008)).
---------------------------------------------------
D09-1107:5	51:233	The approach most closely resembling ours is the independently developed global discriminative loglinear model based on synchronous context-free grammars (Blunsom and Osborne, 2008; Blunsom et al., 2008).
---------------------------------------------------
E09-1088:6	7:222	1 Introduction When data have distinct sub-structures, models exploiting latent variables are advantageous in learning (Matsuzaki et al., 2005; Petrov and Klein, 2007; Blunsom et al., 2008).
---------------------------------------------------
W09-1114:7	10:171	We can also use Bayesian inference techniques to avoid resorting to heuristics that damage the probabilistic interpretation of the models (Zhang et al., 2008; DeNero et al., 2008; Blunsom et al., 2009).
---------------------------------------------------
W09-1114:8	164:171	Approaches to the former (Blunsom et al., 2008; May and Knight, 2006) rely on dynamic programming techniques which do not scale well without heuristic approximations, while approaches to the latter (Smith and Eisner, 2006; Zens et al., 2007) use biased k-best approximations.
---------------------------------------------------
W09-1114:9	116:171	This is almost certainly a side effect of the MERT training approach that was used to construct the models so as to maximise the performance of the model on its single best derivation, without regard to the shape of the rest of the distribution (Blunsom et al., 2008).
---------------------------------------------------
N09-1025:10	23:173	Others have introduced alternative discriminative training methods (Tillmann and Zhang, 2006; Liang et al., 2006; Turian et al., 2007; Blunsom et al., 2008; Macherey et al., 2008), in which a recurring challenge is scalability: to train many features, we need many train218 ing examples, and to train discriminatively, we need to search through all possible translations of each training example.
---------------------------------------------------
E09-1037:11	12:224	Meanwhile, some learning algorithms, like maximum likelihood for conditional log-linear models (Lafferty et al., 2001), unsupervised models (Pereira and Schabes, 1992), and models with hidden variables (Koo and Collins, 2005; Wang et al., 2007; Blunsom et al., 2008), require summing over the scores of many structures to calculate marginals.
---------------------------------------------------
N09-1049:12	32:210	Extensions to Hiero Several authors describe extensions to Hiero, to incorporate additional syntactic information (Zollmann and Venugopal, 2006; Zhang and Gildea, 2006; Shen et al., 2008; Marton and Resnik, 2008), or to combine it with discriminative latent models (Blunsom et al., 2008).
---------------------------------------------------
D08-1066:13	180:243	(Blunsom et al., 2008)).
---------------------------------------------------
D08-1066:14	37:243	Coming up-to-date, (Blunsom et al., 2008) attempt a related estimation problem to (Marcu and Wong, 2002), using the expanded phrase pair set of (Chiang, 2005a), working with an exponential model and concentrating on marginalizing out the latent segmentation variable.
---------------------------------------------------
D08-1066:15	236:243	Secondly, as (Blunsom et al., 2008) show, marginalizing out the different segmentations during decoding leads to improved performance.
---------------------------------------------------
P09-1065:16	217:227	(2008) and Blunsom et al.
---------------------------------------------------
E09-1061:17	165:222	item form: [i,j,ueve] goal: [I,j,ue] rules:     [i,j,ue] R(fifiprime/ejejprime) [iprime,j,ejejprime] [i,j,ueejve] [i,j + 1,ueejve] ej+1 = rj+1 (Logic MONOTONE-ALIGN) Under the boolean semiring, this (minimal) logic decides if a training example is reachable by the model, which is required by some discriminative training regimens (Liang et al., 2006; Blunsom et al., 2008).
---------------------------------------------------
E09-1061:18	159:222	Alignment is often used in training both generative and discriminative models (Brown et al., 1993; Blunsom et al., 2008; Liang et al., 2006).
---------------------------------------------------
E09-1061:19	103:222	We might also want to calculate the total probability of all possible derivations, which is useful for parameter estimation (Blunsom et al., 2008).
---------------------------------------------------
D08-1024:20	15:186	There has been much work on improving MERTs performance (Duh and Kirchoff, 2008; Smith and Eisner, 2006; Cer et al., 2008), or on replacing MERT wholesale (Turian et al., 2007; Blunsom et al., 2008).
---------------------------------------------------
