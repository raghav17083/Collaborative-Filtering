1 Introduction In current troubleshooting spoken dialogue systems (SDS)(Williams, 2007), the major part of the conversation is directed by the system, while the user follows the systems instructions.


As in Williams and Young (2007), we incorporate both the user goal (i.e., the intended product) and the user action in the system state.


Following nota105 tions in Williams and Young (2007), a POMDP is defined as a tuple {S,A,T,R,O,Z,,vectorb0} where S is a set of states s describing the environment; A is a set of machine actions a operating on the environment; T defines a transition probability P(sprime|s,a); R defines a reward function r(s,a); O is a set of observations o, and an observation can be thought as a corrupted version of a user action; Z defines an observation probability P(oprime|sprime,a);  is a geometric discount factor; andvectorb0 is an initial belief vector.


User Knowledge Model: In most of the applications (Roy et al., 2000; Williams, 2007) where 107 the POMDP framework got applied, it is normally assumed that the user needs only common sense to answer the questions asked by the dialog system.


3.2 Applying POMDP in Practice As mentioned in Williams and Young (2007), it is not trivial to apply the POMDP framework to a specific application.


