3 Comparing several parsers for the XTAG grammar In this section, we make a comparison of several different TAG parsing algorithms  the CYKbased algorithm described at (Vijay-Shanker and Joshi, 1985), Earley-based algorithms with (Alonso et al. , 1999) and without (Schabes, 1994) the valid prefix property (VPP), and Nederhofs algorithm (Nederhof, 1999)  on the XTAG English grammar (release 2.24.2001), by using our system and the ideas we have explained.


2 Generating parsers for the XTAG grammar By using parsing schemata as the ones in (Alonso et al. , 1999; Nederhof, 1999) as input to our system, we can easily obtain efficient implementations of several TAG parsing algorithms.


