This extended sharing is similar to the packing approach employed in the Core Language Engine (Alshawi, 1992), except that the CLE relies on a quasi-logical form to underspecify semantic information such as quantifier scope (the calculation of which is deferred until syntactic ambiguities have been at least partially resolved by other means); whereas the approach described here extends structure sharing to incorporate a certain amount of quantifier scope ambiguity in order to allow a complete evaluation of all subderivations in a shared forest before making any disambiguation decisions in syntax.3 Various synchronous formalisms have been introduced for associating syntactic representations with logical functions in isomorphic or locally non-isomorphic derivations, including Categorial Grammars (CGs) (Wood, 1993), Synchronous Tree Adjoining Grammars (TAGs) (Joshi, 1985; Shieber and Schabes, 1990; Shieber, 1994), and Synchronous Description Tree Grammars (DTGs) (Rambow et al. , 1995; Rambow and Satta, 1996).


Next, we prove that in GMTG each component grammar retains its generative power, a requirement for synchronous formalisms that Rambow and Satta (1996) called the weak language preservation property. Lastly, we propose a synchronous generalization of Chomsky Normal Form, which lays the groundwork for synchronous parsing under GMTG using a CKYstyle algorithm (Younger, 1967; Melamed, 2004).


Lemma 2 For any a17 -GMTG a117, there exists a properly synchronous a17 -GMTG a117 a79 such that a145 a5 a117 a79a44a9a43a42 a145 a5 a117 a9, a113 a5 a117a57a79a37a9a139a42 a150a43a152a146a153 a44 a113 a5 a117 a9a11a1 a17a114a56, and a24 a5 a117a57a79a44a9a120a42 a4a146a131a146a70 a5 a24 a5 a117 a9 a9 . Combining Lemmas 1 and 2, we have Theorem 2 For any a17 -GMTG a117, there exists some LCFRS a117 a79 with a145 a5 a117a57a79a44a9 a42 a145 a5 a117 a9 and a113 a5 a117a57a79a37a9 a42 a150a43a152a146a153 a44 a113 a5 a117 a9a11a1 a17 a56 such that a24 a5 a117 a79a44a9a53a42 a5a7a6a10a8a146a72 a5 a4a146a131a146a70 a5 a24 a5 a117 a9 a9 a9 . 5 Weak Language Preservation Property GMTGs have the weak language preservation property, which is one of the defining requirements of synchronous rewriting systems (Rambow and Satta, 1996).


Much of theoretical linguistics can be formulated in a very natural manner as stating correspondences (translations) between layers of representation structures (Rambow & Satta, 1996).


SYNCHRONOUS SSTC STRUCTURE Much of theoretical linguistics can be formulated in a very natural manner as stating correspondences (translations) between layers of representation structures (Rambow & Satta, 1996), such as the relation between syntax and semantic.


