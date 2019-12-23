## Non-trivial simplification
- Covering given classes is equivalent to covering all subclasses of them
  - A cluster containing samples of multiple classes should be divided into multiple subclasses, but they share the same classifier

## Clear formulation
### Inputs
- $N$ subclasses in total
  - $M$ target subclasses, denoted $\mathcal{A} = \{A_i\}$
  - $N-M$ dummy subclasses, denoted $\bar{\mathcal{A}} = \{A_i\}$
- $K$ classifiers, denoted $\mathcal{D} = \{d_j\}$
- $L$ class labels, denoted $\{l_j\}$
- $f(A_i, d_j) := 0, 1$ the classifier certificate that $A_i$ is the positive label in classifier $d_j$
- $g(A_i) := l_j$ the partial map from $A_i$ to label $l_j$
- $score(d_j)$ the accuracy(?) of classifier $d_j$

### Outputs
  - $D \subset \mathcal{D}$ a set of $m$ classifiers "covering" all $M$ target subclasses
    - i.e., it determines whether a sample solely belongs to
      - a subset (with consistent labels) of the $M$ subclasses, or,
      - the rest dummy subclasses $\bar{\mathcal{A}}$
  - $score(D)$ the overall accuracy(?) of the classifier set

### Instance
  - Given constant $k$ and inputs, does there exist an output $D$ with size $m < k$?

## This is an NP problem
- Given certificate $D \subset \mathcal{D}$
- Encode each subclass with a binary string $s$
    - $j$-th element for $i$ subclass is $f(A_i, d_j)$
  - $O(Nk) = O(NK)$
- The algorithm returns $yes$ iff.
  - No two subclasses in $\mathcal{A}$ share the same encoding but have different labels, and,
  - No subclass in $\mathcal{A}$ share the same encoding with any subclass in $\bar{\mathcal{A}}$
  - $O(N)$ using hash table
- Otherwise, it returns $no$
<!-- - Determining feasibility is also polynomial time, if we set the $D$ to be the entire $\mathcal{D}$ -->
