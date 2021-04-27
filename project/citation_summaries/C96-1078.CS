W05-1201:1	80:136	3.2 Automatic alignment Our tree alignment algorithm is based on the dynamic programming algorithm in (Meyers et al. , 1996), and similar to that used in (Barzilay, 2003).
---------------------------------------------------
W05-1201:2	17:136	Initially work focused on word-based alignment, but more and more work is also addressing alignment at the higher levels (substrings, syntactic phrases or trees), e.g., (Meyers et al. , 1996), (Gildea, 2003).
---------------------------------------------------
C00-1078:3	11:146	In our Machine %'anslation system, transfer rules are generated automatically from parsed parallel text along the lines of (Matsulnoto el; al,, 1993; Meyers et al. , 1996; Meyers et al. , 1998b).
---------------------------------------------------
W01-1511:4	20:203	In particular, for MT, we expect it will benefit procedures which learn translation rules from syntactically analyzed parallel corpora, such as (Matsumoto et al. , 1993; Meyers et al. , 1996).
---------------------------------------------------
P04-1083:5	162:203	This kind of synchronizer stands in contrast to more ad-hoc approaches (e.g. , Matsumoto, 1993; Meyers, 1996; Wu, 1998; Hwa et al. , 2002).
---------------------------------------------------
C02-1010:6	22:202	1 Related Works Most of the previous works conduct structure alignment with complex, hierarchical structures, such as phrase structures (e.g. , Kaji, Kida & Morimoto, 1992), or dependency structures (e.g. , Matsumoto et al. 1993; Grishman, 1994; Meyers, Yanharber & Grishman 1996; Watanabe, Kurohashi & Aramaki 2000).
---------------------------------------------------
C02-1010:7	26:202	Then the correspondent structures in different languages are aligned (e.g. , Kaji, Kida & Morimoto 1992; Matsumoto et al. 1993; Grishman 1994; Meyers, Yanharber & Grishman 1996; Watanabe, Kurohashi & Aramaki 2000).
---------------------------------------------------
P98-2139:8	6:157	Other systems, including our own ((Grishman, 1994) and (Meyers et al. , 1996)), syntactically analyze sentences (parse) before acquiring transfer rules (cf.
---------------------------------------------------
P98-2139:9	48:157	(Meyers et al. , 1996)) were designed to produce alignments which preserve the least common ancestor relationship: If nodes a and b map into nodes a' = f(a) and b' = f(b), then f(LCA(a,b)) = LCA(f(a), f(b)) = LCA(a', b').
---------------------------------------------------
W05-1612:10	96:183	3.1 Tree alignment algorithm The tree alignment algorithm is based on [Meyers et al. , 1996], and similar to that used in [Barzilay, 2003].
---------------------------------------------------
W05-1612:11	117:183	The alignment algorithm is tested with the following NODEMATCH function: NODEMATCH(v,vprime) = 8> >>> >< >>>> >: 10 if STR(v) = STR(vprime) 5 if LABEL(v) = LABEL(vprime) 2 if LABEL(v) is a synonym hyperonym or hyponym of LABEL(vprime) 0 otherwise 1In the original formulation of the algorithm by [Meyers et al. , 1996], there is a penalty for skipping edges.
---------------------------------------------------
W07-1414:12	119:133	and with paraphrasing (+) Task Dev??Dev+ Test??Test+ IE 59.5 61.0 53.0 53.5 IR 67.0 68.0 58.5 61.5 QA 76.0 76.5 69.0 68.0 SUM 66.0 67.5 53.0 53.5 Overall 66.9 68.2 58.6 59.1 of our RTE2 system as described in (ref supressed for blind reviewing) The core of the system is still the tree alignment algorithm from (Meyers et al. , 1996), but without normalization of node weights and applied to Minipar instead of Maltparser output.
---------------------------------------------------
W03-1602:13	128:182	4.3 Three-layered representation Previous work on transfer-based MT systems (Lavoie et al. , 2000; Dorna et al. , 1998) and alignment-based transfer knowledge acquisition (Meyers et al. , 1996; Richardson et al. , 2001) have proven that transfer knowledge can be best represented by declarative structure mapping (transforming) rules each of which typically consists of a pair of source and target partial structures as in the middle of Figure 2.
---------------------------------------------------
W03-1602:14	122:182	4.2 Dependency trees (MDSs) Previous work on transfer-based machine translation (MT) suggests that the dependency-based representation has the advantage of facilitating syntactic transforming operations (Meyers et al. , 1996; Lavoie et al. , 2000).
---------------------------------------------------
