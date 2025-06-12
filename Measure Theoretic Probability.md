https://staff.fnwi.uva.nl/p.j.c.spreij/onderwijs/master/mtp.pdf

## 1. $\sigma$-Algebras and measures

For any non-empty set $S$ we can select some collection of subsets $\Sigma_0 \subset 2^S$.
Such a collection of subsets is called an *algebra* on S if:
	(i) $S \in \Sigma_0$,
	(ii) $E \in \Sigma_0 \Rightarrow E^c \in \Sigma_0$,
	(iii) $E, F \in \Sigma_0 \Rightarrow E \cup F \in \Sigma_0$.
We notice immediately that the empty set $\emptyset$ always belongs to any algebra, since algebras contain the original set $S$ by (i) and are closed under complements (ii), thus the empty set, as the complement of the original set, must be member of any algebra. 

Property (iii) extends to finite unions by induction.

Furthermore, we can show that property (iii) also implies closure under finite intersections and finite set differences, as can be shown:
	$E \cap F=\left(E^c \cup F^c\right)^c$ 
	Leading to:
	$E, F \in \Sigma_0 \Rightarrow E \cap F \in \Sigma_0$,
	and analogously:
	$E \backslash F=E \cap F^c \in \Sigma_0$

### 1.2 Definition of $\sigma$-Algebra

For any nonempty set $S$, a collection of subsets $\Sigma \subset 2^S$ is called a $\sigma$-Algebra on S if it is an algebra and closed under countable unions: 
	$\bigcup_{n=1}^{\infty} E_n \in \Sigma$ for any $E_n \in \Sigma$ $(n = 1, 2, \ldots)$.

A pair $(S, \Sigma)$ is called a measurable space. Elements of $\Sigma$ are called measurable sets. 

#### 1.2.1 $\sigma$-Operator
For any collection $\mathcal{C}$ of subsets of a set $S$ we define the $\sigma$-Operator $\sigma(\mathcal{C})$. This Operator yields the smallest $\sigma$-Algebra that contains $\mathcal{C}$. $\sigma(\mathcal{C})$ is therefore the intersection of all $\sigma$-algebras that contain $\mathcal{C}$. This is very similar to the way we define inner and outer measures in Real Analysis, where we defined the smallest open cover of an interval as the intersection of all open covers of the respective interval. 
We say that $\mathcal{C}$ generates $\Sigma$. 
The union of two $\sigma$-algebras is not necessarily a $\sigma$-algebra. We write
$\Sigma_1 \vee \Sigma_2$ for $\sigma\left(\Sigma_1 \cup \Sigma_2\right)$ instead.

#### 1.2.2 Borel sets and the Borel $\sigma$-algebra
Let $\mathcal{O}$ be the collection of all open subsets of $\mathbb{R}$ with the usual topology (in which all intervals (a, b) are open). Then we define
$\mathcal{B}:=\sigma(\mathcal{O})$.
As the Borel $\sigma$-Algebra. This notion can get quite abstract for sets in the general sense, but for the real numbers we can construct $\mathcal{B}$ like so: 

### Proposition 1.3 Construction of $\mathcal{B}$

Let $\mathcal{I}=\{(-\infty, x]: x \in \mathbb{R}\}$. Then $\sigma(\mathcal{I})=\mathcal{B}$.
**Proof:** First we show that any half open interval $(-\infty, x]$ can be expressed as a countable union of open intervals: 
	$(-\infty, x]=\cap_n\left(-\infty, x+\frac{1}{n}\right)$ 
Because $\sigma$-Algebras are closed under countable unions and thus, by De Morgan, closed under countable intersections we have shown that $\mathcal{I} \subset \mathcal{B}$ and $\sigma(\mathcal{I}) \subset \mathcal{B}$ since $\sigma(\mathcal{I})$ is the smallest $\sigma$-Algebra that contains $\mathcal{I}$. 
For the reverse inclusion we first observe that any open interval, by the definition of openness, can be expressed as a countable union of half open intervals like so: 
	$(-\infty, x)=\cup_n\left(-\infty, x-\frac{1}{n}\right] \in \sigma(\mathcal{I})$.
Thus, we can express any open interval $(a, b)$ as:
	$(a, b)=(-\infty, b) \backslash(-\infty, a] \in \sigma(\mathcal{I})$.
Where the inclusion in $\sigma(\mathcal{I})$ holds since algebras are closed under set difference.
We can now proceed by using the density, and countability, of the Rational numbers as follows: 
Let $G$ be an arbitrary open set.
By openness there must exist a rational number $\varepsilon_x>0$ for every $x \in G$ such that the open interval $\left(x-2 \varepsilon_x, x+2 \varepsilon_x\right) \subset G$. 
Now, considering the open interval $\left(x-\varepsilon_x, x+\varepsilon_x\right)$ nested in our original interval, we can pick, by density of the Rationals, any rational $q_x$ in this interval. For any such $q_x$ it holds that: $\left|x-q_x\right| \leq \varepsilon_x$.
We can see that: 
	$x \in\left(q_x-\varepsilon_x, q_x+\varepsilon_x\right) \subset \left(x-2 \varepsilon_x, x+2 \varepsilon_x\right) \subset G$
Meaning, that we can express our original $G$ as a countable union of such open intervals: 
	$G \subset \cup_{x \in G}\left(q_x-\varepsilon_x, q_x+\varepsilon_x\right) \subset G$
Where the union is indeed countable, because there exist only countably many rational points $q_x$ and $\varepsilon_x$.
This means in turn, that $G$ is indeed part of the $\sigma$-Algebra of open intervals: 
	$G \in \sigma(\mathcal{I})$  
Which leads us to the desired conclusion:
	$\mathcal{O} \subset \sigma(\mathcal{I})$ and therefore
	$\mathcal{B} \subset \sigma(\mathcal{I})$. (Since $\mathcal{B}$) was defined as the smallest such $\sigma$-Algebra. $\square$

Does any subset of $\mathbb{R}$ fall into $\mathcal{B}$? 
No. Vitali sets are a popular counter example, usually employed in the context of the measure problem, to show that there cannot exist a general measure function that assigns any subset of $\mathbb{R}$ a valid measure. One can show that the cardinality of $\mathcal{B}(\mathbb{R})$ is the same as the cardinality of $\mathbb{R}$ itself, leading to the same conclusion. I skip this proof here, please refer to the linked text for the full proof. The general idea is to show that both the cardinality of $\mathcal{B}(\mathbb{R})$ and the cardinality of the reals are equivalent (intuitively because the reals have the same cardinality as the power set of the natural numbers/the set of all infinite binary sequences, and we generate all substs of $\mathcal{B}(\mathbb{R})$ with a countable number of operations). Thus, when one shows that both of these sets have the same cardinality we notice that the powerset of the reals is of strictly larger cardinality $2^{\mathfrak{c}}$ which implies that there must be some subsets of the reals that are not contained in $\mathcal{B}(\mathbb{R})$.

### 1.2 Measures

In the context of probability we're interested in set functions that assign any subset of a sigma algebra a number ("measurement"). This function should also adhere to certain restrictions.

Let $S$ be any set and $\Sigma$ any $\sigma$-Algebra on $S$. A mapping $\mu: \Sigma \rightarrow[0, \infty]$ is 
	(i) *finitely additive* if $\mu(\emptyset)=0$ and $\mu(E \cup F)=\mu(E)+\mu(F)$ for every pair of disjoint sets $E$ and $F$.
	(ii) *$\sigma$-additive* if $\mu(\emptyset)=0$ and if $\mu\left(\cup_n E_n\right)=\sum_n \mu\left(E_n\right)$ for any sequence of disjoint sets of $\Sigma$ for which its union is also in $\Sigma$. (Which is technically fulfilled by additivity and closure under countable unions already).

