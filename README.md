Codebase to analyze a 2x2 quantum puzzle as described in arxiv_link. 

Permutation_universality.ipynb contains a notebook which investigates various questions of universality of the 2x2 quantum puzzle. We explicitly implement the algorithm described in Sawicki and Karnas (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062303). The relevant function are found in Universality_check.py. Gate_helper.py implements some useful functions for generating common qudit gates in arbitrary dimension.

Permutation_strategies.ipynb contains an analysis of solving the 2x2 puzzle using the rules from arxiv_link. Here a set of random scrambles are generated and solved using classical computation. Any raw solving data from these montecarlo samples is saved in Data.
