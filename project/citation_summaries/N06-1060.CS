This approach will generally take advantage of language-specific (e.g. in (Freeman et al., 2006)) and domain-specific knowledge, of any external resources (e.g. database, names dictionaries, etc.), and of any information about the entities to process, e.g. their type (person name, organization, etc.), or internal structure (e.g. in (Prager et al., 2007)).


In Table 1, the MALINE row 3  shows that the English name has a palato-alveolar modification   2 As (Freeman et al., 2006) point out, these insights are not easy to come by: These rules are based on first author Dr. Andrew Freemans experience with reading and translating Arabic language texts for more than 16 years (Freeman et al., 2006, p. 474).


2 Basic Approaches 2.1 Cross-Lingual Approach Our cross-lingual approach (called MLEV) is based on (Freeman et al. 2006), who used a modified Levenshtein string edit-distance algorithm to match Arabic script person names against their corresponding English versions.


For this study, the Levenshtein edit-distance score (where a perfect match scores zero) is  Roman Chinese (Pinyin) Alignment Score LEV ashburton ashenbodu |   a   s   h   b   u   r   t   o   n   | |   a   s   h   e   n   b  o  d    u   | 0.67 MLEV ashburton ashenbodu |  a   s   h       b   u   r    t   o   n  | |  a   s   h   e   n   b   o     d   u    | 0.72 MALINE asVburton aseCnpotu |   a   sV    b   <   u   r   t   o   |   n |   a   s   eC  n   p   o     t   u   |   0.48 3 normalized to a similarity score as in (Freeman et al. 2006), where the score ranges from 0 to 1, with 1 being a perfect match.


