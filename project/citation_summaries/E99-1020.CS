W06-1514:1	86:133	3 Comparing several parsers for the XTAG grammar In this section, we make a comparison of several different TAG parsing algorithms  the CYKbased algorithm described at (Vijay-Shanker and Joshi, 1985), Earley-based algorithms with (Alonso et al. , 1999) and without (Schabes, 1994) the valid prefix property (VPP), and Nederhofs algorithm (Nederhof, 1999)  on the XTAG English grammar (release 2.24.2001), by using our system and the ideas we have explained.
---------------------------------------------------
W06-1514:2	45:133	2 Generating parsers for the XTAG grammar By using parsing schemata as the ones in (Alonso et al. , 1999; Nederhof, 1999) as input to our system, we can easily obtain efficient implementations of several TAG parsing algorithms.
---------------------------------------------------
W06-1514:3	19:133	The parsing schema for the TAG CYK-based algorithm (Alonso et al. , 1999) is a function that maps such a grammar G to a deduction system whose domain is the set of items {[N,i,j,p,q,adj]} verifying that N is a tree node in an elementary 1Where VT denotes the set of terminal symbols, VN the set of nonterminal symbols, S the axiom, I the set of initial trees and A the set of auxiliary trees.
---------------------------------------------------
W06-1514:4	87:133	The schemata for these algorithms without unification support can be found at (Alonso et al. , 1999).
---------------------------------------------------
P08-1110:5	11:218	(Alonso et al., 1999).
---------------------------------------------------
