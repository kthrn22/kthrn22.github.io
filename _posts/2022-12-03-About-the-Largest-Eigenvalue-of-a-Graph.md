---
title: About the Largest Eigenvalue of a Graph
date: 2022-12-03
permalink: /posts/About the Largest Eigenvalue of a Graph
tags:
  - Algebraic Graph Theory
---

# About the Largest Eigenvalue of a Graph


## Table of Contents

* [Rayleigh Quotient](#rayleigh-quotient)

* [Rayleigh - Ritz](#rayleigh---ritz-theorem)

* [Cauchy Interlace theorem](#cauchy-interlace-theorem)

* [Largest Eigenvalue](#largest-eigenvalue)

* [Wilf's Theorem](#wilfs-theorem)

* [Perron-Frobenius Theorem for Symmetric Matrices](#perron-frobenius-theorem-for-symmetric-matrices)
# Rayleigh Quotient

Let $\mathbf{A}$ be a real symmetric matrix, $\mathbf{x}$ be a non-zero vector, the Rayleigh Quotient of $\mathbf{x}$ with respect to $\mathbf{A}$ is defined as 

$$ \frac{\mathbf{x}^{T}\mathbf{A}\mathbf{x}}{\mathbf{x}^T \mathbf{x}} $$

If $\mathbf{\lambda}$ is an eigenvalue of $\mathbf{A}$  and $\mathbf{u}$ is the corresponding eigenvector then $\frac{\mathbf{u}^T \mathbf{A} \mathbf{u}}{\mathbf{u}^T \mathbf{u}} = \mathbf{\lambda}$ 

# Rayleigh - Ritz theorem

If $\mathbf{A}$ is a real symmetric $n \times n$ matrix with eigenvalues $\lambda_1 \leq \lambda_2 \leq ... \leq \lambda_n$ and a set of orthonormal eigenvectors $(\mathbf{u}_1, \mathbf{u}_2, ..., \mathbf{u}_n)$ then 

$$
\max_{\mathbf{x} \neq 0, \mathbf{x} \in \{\mathbf{u}_1, ..., \mathbf{u}_i\}} \frac{\mathbf{x}^T \mathbf{A} \mathbf{x}}{\mathbf{x}^T \mathbf{x}} = \lambda_i, ~\min_{\mathbf{x} \neq 0, \mathbf{x} \in \{\mathbf{u}_1, ..., \mathbf{u}_{i - 1}\}^{\bot}} \frac{\mathbf{x}^T \mathbf{A} \mathbf{x}}{\mathbf{x}^T \mathbf{x}} = \lambda_i
$$

### Proof

For any $\mathbf{x} \in \{\mathbf{u}_1, \mathbf{u}_2, ..., \mathbf{u}_i \}$, $\mathbf{x}$ is a linear combination of $\mathbf{u}_1, ..., \mathbf{u}_i$. Thus we can write $\mathbf{x}$ as $\mathbf{x} = c_1\mathbf{u}_1 + ... + c_i\mathbf{u}_i = \mathbf{U}\mathbf{c} ~(c_1, ..., c_i \in \mathbb{R})$, where $\mathbf{U} \in \mathbb{R}^{n \times i}$ has $\mathbf{u}_j$ as its $j$-th column and $\mathbf{c} \in \mathbb{R}^{i \times 1}$ with $c_1, ..., c_i$ as its entries.

$$
\begin{split} \frac{\mathbf{x}^T \mathbf{A} \mathbf{x}}{\mathbf{x}^T \mathbf{x}} &= \frac{\mathbf{(Uc)^\intercal A (Uc)}}{\mathbf{(Uc)^\intercal(Uc)}} \\ & = \frac{\mathbf{c^ \intercal U^\intercal A Uc}}{\mathbf{c^\intercal U^\intercal U c}} \\ &= \frac{\mathbf{c^\intercal D c}}{\mathbf{c^\intercal c}} ~(\mathbf{U^ \intercal U = I, D} \text{ is a diagonal matrix has } \lambda_1, ..., \lambda_i \text{ as its entries}) \\ &= \frac{\sum_{k = 1}^{i} \lambda_kc_k^2}{\sum_{k = 1}^{i} c_k^2} \leq \frac{\lambda_i (\sum_{k = 1}^{i} c_k^2)}{\sum_{k = 1}^{i} c_k^2} = \lambda_i ~(\lambda_1 \leq ... \leq \lambda_i)                     \end{split}
$$

Similarly

$$
\frac{\mathbf{x^\intercal A x}}{\mathbf{x^\intercal x}} \geq \lambda_i ~\text{for } \mathbf{x} \neq 0, x \in \mathbf{x} \in \{\mathbf{u}_1, ..., \mathbf{u}_{i - 1}\}^{\bot} 
$$


# Cauchy Interlace theorem

For 2 sequences of real numbers: $a_1 \leq a_2 \leq ... \leq a_n$ and $b_1 \leq b_2 \leq ... \leq b_m$ with $m < n$. The 2nd sequence interlace the first sequence when:

$$
a_i \leq b_i \leq a_{n - m + i} ~\text{for } i = 1,2, ..., m
$$

Let $\mathbf{A}$ be a $n \times n$  real symmetric matrix with eigenvalues $\lambda_1 \leq \lambda_2 \leq ... \leq \lambda_n$ and $\mathbf{P}$ is a $n\times m$ real matrix such that $\mathbf{P^\intercal P = I}$. Let $\mathbf{B = P^\intercal AP}$ $(\mathbf{B} \in m \times m)$ with $m$ eigenvalues $\alpha_1 \leq \alpha_2 \leq ... \leq \alpha_m$ then

- $\mathbf{B}$’s eigenvalues interlace those of $\mathbf{A}$
    
    ### Proof
    
    Let $(\mathbf{a_1, ..., a_n}), (\mathbf{b_1, ..., b_m})$ be the orthonormal sets of eigenvectors of $\mathbf{A}$ and $\mathbf{B}$, respectively. 
    
    For any $i \neq j, \mathbf{(P^\intercal a_i)^\intercal (P^\intercal a_j)=a_i^\intercal P P^\intercal a_j = a_i^\intercal I a_j = a_i^\intercal a_j = 0}$ . Thus $(\mathbf{P^\intercal a_1,..., P^\intercal a_n})$ is orthogonal.
    
    For $i = 1, 2, ..., m$:
    
    Let $T = \{\mathbf{b_1, ..., b_i}\} \Rightarrow \dim(T) = i$ and $S = \{\mathbf{P^\intercal a_i, ..., P^\intercal a_n}\} \Rightarrow \dim(S) = n - i + 1$ (since $(\mathbf{P^\intercal a_i, ..., P^\intercal a_n})$ is orthogonal). 
    
    Suppose that $T \cap S =\emptyset$. Thus, $\dim(T \cup S) = (i) + (n - i + 1) = n + 1$, but $\dim(T \cup S) \leq m$ (since each element is a $m \times 1$ vector) and hence we have a contradiction. Therefore, $T \cap S \neq \emptyset$, and we can choose a non-zero vector $\mathbf{q} \in T \cap S (q \in \mathbb{R}^{m \times 1})$.
    
    $\mathbf{q} \in T \cap S \Rightarrow \mathbf{q} \in \{\mathbf{b_1, ..., b_i}\}$. By Rayleigh-Ritz theorem, we have
    
    $$
    \begin{equation} \frac{\mathbf{q^\intercal B q}}{\mathbf{q^\intercal q}} \leq \alpha_i \end{equation}
    $$
    
    $\mathbf{q} \in \mathbf{\{P^\intercal a_i, ..., P^\intercal a_n\}} \Rightarrow \mathbf{Pq} \in \{\mathbf{a_i, ..., a_n}\}$. By Rayleigh-Ritz theorem, we have
    
    $$
    \begin{equation} \frac{\mathbf{q^\intercal B q}}{\mathbf{q^\intercal q}} = \frac{\mathbf{(Pq)^\intercal A(Pq)}}{\mathbf{(Pq)^\intercal(Pq)}} \geq \lambda_i \end{equation}
    $$
    
    From $(1), (2) \Rightarrow \lambda_i \leq \frac{\mathbf{q^\intercal B q}}{\mathbf{q^\intercal q}} \leq \alpha_i$
    
    Similarly, we can choose a non-zero vector $\mathbf{q}$ from $\{\mathbf{b_i, ..., b_n}\} \cup \{\mathbf{P^\intercal a_1, ..., P^\intercal a_{n - m + i}}\}$ and prove that $\alpha_i \leq \frac{\mathbf{q^\intercal B q}}{\mathbf{q^\intercal q}} \leq \lambda_{n - m + 1}$ using Rayleigh-Ritz theorem. 
    
    Therefore, we have proven that $\lambda_i \leq \alpha_i \leq \lambda_{n - m + i}$
    
- If $\lambda_i = \alpha_i ~(1 \leq i \leq m)$ then $\mathbf{B}$ has an eigenvector $\mathbf{u}$ with eigenvalue $\alpha_i$ such that $\mathbf{Pu}$ is an eigenvector of $\mathbf{A}$ with eigenvalue $\lambda_i$
    
    ### Proof
    
    $$
    \begin{split} \mathbf{Bu = \alpha_iu} &\Leftrightarrow \mathbf{P^\intercal APu = \lambda_iu}\\ &\Leftrightarrow \mathbf{P(P^\intercal A P u) = \lambda_i (Pu)} \\ &\Leftrightarrow (\mathbf{PP^\intercal) APu} = \lambda_i (\mathbf{Pu}) \\ &\Leftrightarrow \mathbf{A(Pu) = \lambda_i (Pu)}\end{split}
    $$
    
    Thus, $\mathbf{Pu}$ is an eigenvector of $\mathbf{A}$ with eigenvalue $\lambda_i$
    

## **Corollary**

If a graph $G$ with adjacency matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ and eigenvalues $\lambda_1 \geq...\geq \lambda_n$, $H$ is $G$’s induced subgraph with adjacency matrix $\mathbf{B} \in \mathbb{R}^{m \times m}$ and eigenvalues $\mu_1 \geq ... \geq \mu_m$ ($m \leq n$) then

$$
\lambda_{n-m+i} \leq \mu_i \leq \lambda_{i}
$$

### Proof

Since $H$ is $G$’s induced matrix, so $\mathbf{B}$ is a principal sub-matrix of $\mathbf{A}$, .i.e $\mathbf{B}$ can be obtained by deleting the same row and column of $\mathbf{A}$. Specifically if $\mathbf{P} \in \mathbb{R}^{m \times n}$  has entries is $0$ or $1$ and each row has exactly one $1$ entry, each column has at most one $1$ entry. Then $\mathbf{B = P^\intercal AP}$ and $\mathbf{P^\intercal P = I}$ . Therefore, the corollary can be derived from the Interlacing Theorem. 

# Largest Eigenvalue

For a graph $G$, denote is adjacency matrix as $\mathbf{A}$ , $\mu_1$ as $\mathbf{A}$’s largest eigenvalue, and $\bar{d}$ as $G$’s average degree, then

$$
\bar{d} \leq \mu_1
$$

### Proof

Let $\mathbf{x} = \mathbf{1}, \mathbf{x} \in \mathbb{R}^{n \times 1}$ ($x$ is all-one vector), and $\mathbf{A}$ is the adjacency matrix of $G$ ($\mathbf{A} \in \mathbb{R}^{n \times n }$). Thus, from Rayleigh - Ritz Theorem

$$
\mathbf{\frac{x^\intercal A x}{x^\intercal x}} \leq \mu_1
$$

Since $\mathbf{x = 1} \Rightarrow \mathbf{(Ax)}_i$ (the i-th entry of $\mathbf{Ax}$) is the degree of vertex $i$, $d_i$. Thus, $\mathbf{x^\intercal(Ax)} = \sum_{i = 1}^n d_i$. Moreover, $\mathbf{x^\intercal x} = n$, so

$$
\mathbf{\frac{x^\intercal Ax}{x^\intercal x}} = \frac{\sum d_i}{n} = \bar{d} \leq \mu_1 
$$

Denote $d_{max}$ as $G$’s largest degree, then

$$
\mu_1 \leq d_{max}
$$

### Proof

Denote the eigenvector corresponds to $\mu_1$ as $\mathbf{u}$, and suppose that on vertex $a$, $\mathbf{u}$ takes the maximum value, .i.e $\mathbf{u}(a) \geq \mathbf{u}(v)$ for all vertex $v$ in graph $G$. As $\mathbf{Au} = \mu_1\mathbf{u}$, so $\mu_1 \mathbf{u}(a) = \mathbf{Au}(a) = \sum_{v \in N(a)} \mathbf{u}(v) \leq d_a \mathbf{u}(a)$, where $N(a)$ is the set of vertices that are adjacnet to $a$ and $d_a$ is vertex $a$’s degree. Without loss of generality, assume that $\mathbf{u}(a) > 0$. Since $\mu_1 \mathbf{u}(a) \leq d_a \mathbf{u}(a)$, so $\mu_1 \leq d_a \leq d_{max}$.


# Wilf’s Theorem

Graph Coloring is an assignment of colors to vertices so that adjacent vertices have different colors. If $k$ colors can be used to color a graph $G$ then $G$ is said to be $k$-colorable. The Chromatic Number of a graph is the minimum number of colors that can be used to color the graph. Denote the Chromatic Number of graph $G$ as $\chi(G)$. 

$$
\chi(G) \leq \lfloor \mu_1\rfloor + 1
$$

where $\mu_1$ is the largest eigenvalue of $G$

### Proof

The theorem can be proven using induction. 

For a graph that has only one vertex and has no edge, it can be colored by one color and its largest eigenvalue is zero. Thus, this graph satisfies the theorem.

Suppose that the theorem is correct for graph that has $n - 1$ vertices. Let $G$ is a graph that has $n$ vertices and its largest eigenvalues is $\mu_1$. Since $\mu_1$ is greater than or equal to the average degree of $G$, so there is a vertex $a$ such that $d_a \leq \lfloor \mu_1 \rfloor$, where $d(a)$ is the degree of vertex $a$. Let $H$ is an induced subgraph of $G$ without vertex $a$, and denote the largest eigenvalue of $H$ as $\lambda_1$, then $\lambda_1 \leq \mu_1$. Since $H$ has $n - 1$ vertices so $\chi(H) \leq \lfloor \lambda_1  \rfloor + 1 \leq \lfloor \mu_1  \rfloor + 1$, implying that $H$ can be colored using at most $\lfloor \mu_1\rfloor + 1$. Suppose such coloring is labeled as $\{1, ..., \lfloor u_1 \rfloor\  + 1\}$. For graph $G$, since $a$ has at most $\lfloor u_1 \rfloor$ neighbors, so there is a color in $\{1, ..., \lfloor u_1 \rfloor\  + 1\}$ that does not appear in $a$’s neigborhood and that color can be assigned to $a$. Thus $G$ can be colored using at most $\lfloor \mu_1 \rfloor + 1$  colors. 

# Perron-Frobenius Theorem for Symmetric Matrices

If $G$ is a connected weighted graph with adjacency matrix $\mathbf{A}$ and eigenvalues $\lambda_1 \geq... \geq \lambda_n$ then 

- The corresponding eigenvector of $\lambda_1$ has strictly positive entries
- The multiplicity of $\mu_1$ is $1$.
- $\lambda_1 \geq |\lambda_n|$

### Lemma

Suppose that $\mathbf{u}$ is not stricly positive, so there is a vertex $a$ such that $\mathbf{u}(a) = 0$. Since $G$ is connected, there is a vertex $b$ that is adjacenet to $a$ such that $\mathbf{u}(b) > 0$ and $\mathbf{A}_{a, b} > 0$. Since $\mathbf{Au}(a) = \sum_{v \in N(a)}\mathbf{A}_{a, v} \mathbf{u}(v) =\lambda_1 \mathbf{u}(a) = 0$. However, since the entries of $\mathbf{A}, \mathbf{u}$ is non-negative and $\mathbf{A}_{a, b} \mathbf{u}(b) > 0$, contributing a positive value to $\sum_{v \in N(a)}\mathbf{A}_{a, v} \mathbf{u}(v)$, which will make $\sum_{v \in N(a)}\mathbf{A}_{a, v} \mathbf{u}(v) =\lambda_1 \mathbf{u}(a) > 0$. Thus, we obtain a contradiction, and the Lemma is proven. 

- The corresponding eigenvector of $\lambda_1$ has strictly positive entries
    
    ### Proof
    
    Denote $\mathbf{u}$ is the eigenvector of $\mu_1$ and vector $\mathbf{x}$ is constructed as following:
    
    $$
    \mathbf{x}(i) = |\mathbf{u}(i)|, \forall i
    $$
    
    Therefore, $\mathbf{x^\intercal x = u^\intercal u}$. Without loss of generality, assume that $\mathbf{x^\intercal x} = 1$ 
    
    $$
    \mu_ 1 = \mathbf{u^\intercal Au} = \sum \mathbf{A}_{ij}\mathbf{u}(i)\mathbf{u}(j) \leq \sum \mathbf{A}_{ij}\mathbf{x}(i)\mathbf{x}(j) = \mathbf{x^\intercal Ax}
    $$
    
    Since $\mathbf{\frac{x^\intercal Ax}{x^\intercal x}} \leq \mu_1$ (Rayleight-Ritz Theorem), so $\mathbf{x^\intercal Ax} = \mu_1$, and $\mathbf{x}$ is a non-negative eigenvector of $\mu_1$. Thus, $\mathbf{x}$ is stricly positive.
    
- The multiplicity of $\mu_1$ is $1$.
    
    ### Proof
    
    Let $\mathbf{u}_1$ be the eigenvector corresponds to $\mu_1$ and $\mathbf{u}_2$ is the eigenvector of $\mu_2$. If the multiplicity of $\mu_1$ is $1$ then we need to prove that $\mu_1 > \mu_2$. 
    
    Suppose that vector $\mathbf{y}$ is constructed as following:
    
    $$
    \mathbf{y}(i) = |\mathbf{u}_2(i)|, \forall i
    $$
    
    Thus,
    
    $$
    \mu_ 2 = \mathbf{u}_2 ^\intercal \mathbf{Au}_2 = \sum \mathbf{A}_{ij}\mathbf{u}_2(i)\mathbf{u}_2(j) \leq \sum \mathbf{A}_{ij}\mathbf{y}(i)\mathbf{y}(j) = \mathbf{y^\intercal Ay}\leq \mu_1
    $$
    
    If $\mu_2 = \mu_1$ then $\mathbf{y}(i) = \mathbf{u}_2(i), \forall i$, so $\mathbf{u}_2(i) \geq 0, \forall i$. As $\mathbf{u}_2$ is orthogonal to $\mathbf{u}_1$ so $\sum_i \mathbf{u}_2(i) \mathbf{u}_1(i) = 0$, but $\mathbf{u}_1(i) > 0$ (derived from Lemma), $\mathbf{u}_2(i) \geq 0, \forall i$. Thus, if $\sum_i \mathbf{u}_2(i) \mathbf{u}_1(i) = 0$ then $\mathbf{u}_2(i) = 0, \forall i$, which is a contradiction. Therefore, $\mu_1 > \mu_2$ and the multiplicity of $\mu_1$ is $1$. 
    
- $\mu_1 \geq |\mu_n|$
    
    ### Proof
    
    Denote $\mathbf{u}_n$ as the eigenvector corresponds to $\mu_n$ and $\mathbf{u}_1$ as the eigenvector of $\mu_1$. Similar to the proof of previous arguments, construct a vector $\mathbf{x}$ as following:
    
    $$
    \mathbf{x}(i) = |\mathbf{u}_n(i)|, \forall i
    $$
    
    $|\mu_n| = |\mathbf{u}_n^\intercal \mathbf{Au}_n | = |\sum \mathbf{A} \mathbf{u}_n(i)\mathbf{u}_n(j)| \leq \sum_{ij}\mathbf{A} \mathbf{x}(i)\mathbf{x}(j) = \mathbf{x^\intercal Ax} \leq \mu_1$
    
    Thus, $\mu_1 \geq |\mu|_n$
    

# References

**Daniel A. Spielman**, [Spectral and Algebraic Graph Theory](http://cs-www.cs.yale.edu/homes/spielman/sagt/sagt.pdf)

****CIS 5150, Fall 2022, University of Pennsylvania,**** [Rayleigh Ratios and the Courant-Fischer Theorem](https://www.cis.upenn.edu/~cis5150/cis515-15-spectral-clust-appA.pdf)
