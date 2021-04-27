Thus given a CPS formalization of the parsing problem and an appropriate memoization technique, it is in fact the case that "the maintenance of well-formed substring tables or charts can be seen as a special case of a more general technique: memoization" (Norvig 1991), even if the grammar contains left recursion.


Memoization is a standard technique for avoiding redundant computation, and as Norvig (1991) noted, it can be applied to top-down recognizers to convert exponentialtime recognizers into polynomial-time recognizers.


Introduction In a paper published in this journal, Norvig (1991) pointed out that memoization of a top-down recognizer program produces a program that behaves similiarly to a chart parser.


The problem of exponential time complexity in top-down parsers constructed as sets of mutuallyrecursive functions has been solved by Norvig (1991) who uses memotables to achieve polynomial complexity.


4.1 Memoization As in Norvig (1991) a memotable is constructed during recognition.


