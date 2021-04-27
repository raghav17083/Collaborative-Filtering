5.1 Markov decision processes We follow past lines of research (such as Levin et al. , 2000; Singh et al. , 2002) by representing a dialogue as a trajectory in the state space, determined 795 by the user responses and system actions: s1 a1,r1 s2 a2,r2sn an,rn sn+1, in which si ai,ri si+1 means that the system performed action ai in state si, received1 reward ri and changed to state si+1.


2 Related Work Previous work has examined learning of effective dialogue strategies for information seeking spoken dialogue systems, and in particular the use of reinforcement learning methods to learn policies for action selection in dialogue management (see e.g. Levin et al. , 2000; Walker, 2000; Scheffler and Young, 2002; Peek and Chickering, 2005; Frampton and Lemon, 2006), for selecting initiative and confirmation strategies (Singh et al. , 2002); for detecting speech recognition problem (Litman and Pan, 2002); changing the dialogue according to the expertise of the user (Maloor and Chai, 2000); adapting responses according to previous interactions with the users (Rudary et al. , 2004); optimizing mixed initiative in collaborative dialogue (English and Heeman, 2005), and optimizing confirmations (Cuayahuitl et al. , 2006).


Many studies have been conducted on efficient dialogue strategies (Walker et al. , 1998; Litman et al. , 2000; Komatani et al. , 2002), but it is not clear how to accomplish a more human-like enthusiasm for a conversational dialogue.


