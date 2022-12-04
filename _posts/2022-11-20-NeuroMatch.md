---
title: Neural Subgraph Matching (NeuroMatch) explained
date: 2022-04-17
permalink: /posts/NeuroMatch
tags:
  - GNN
  - Pooling
---

# Neural Subgraph Matching (NeuroMatch) explained

## What is subgraph matching?

Given a query graph $G_Q$ and a larger target graph $G_T$, subgraph isomorphism matching is a task that determines whether $G_Q$ is isomorphic to any subgraph of $G_T$. In theoretical computer science, this task is NP-complete, existing methods based on combinatorial matching become computationally expensive when it comes to large query and target graphs.

## Overview

NeuroMatch offers a fast, neural approach to subgraph matching by using Graph Neural Networks (GNNs) to capture subgraph relations and embed the query and target graph to vector space.

To predict whether $G_Q$ is isomorphic to a subgraph of $G_T$, the authors of NeuroMatch address the following problem:

Given a neighborhood $G_u$ around node $u$ of $G_T$ and the query graph anchored at node $v$ of $G_Q$, $G_v$, predict whether $G_v$ is a subgraph of $G_u$.

### The Algorithm:

For every node $u$ of $G_Q$ and $v$ of $G_T$:

- Compute embedding of node $u$’s $k$-hop neighborhood ($G_u$), $\mathbf{z}_u$, and embedding of $G_T$ anchored at $v$ ($G_v$), $\mathbf{z}_v$
- Compute the subgraph prediction function $f(\mathbf{z}_u, \mathbf{z}_v)$ (whether $G_v$ is a subgraph of $G_u$)

Compute the average score of all $f(\mathbf{z}_u, \mathbf{z}_v)$ and make prediction base on the score

## Main Ideas

### Embedding Stage

For every node $u$ of the target graph $G_T$, extract its $k$-hop neighborhood and apply $k$-layer GNN on the $k$-hop neighborhood to obtain node embedding $\mathbf{z}_u$. 

For every node $v$ of the query graph $G_Q$, denote the query graph anchored at node $v$ as $G_v$. GNN is applied on $G_v$ to obtain node embedding $\mathbf{z}_v$. 

**Choice of $k$:** Depends on the size of the query graph. Specifically, $k$ must be at least the diameter of $G_Q$.

### Subgraph Relation

**Subgraph relation is PARTIAL ORDER:**

- *Reflexivity*: a graph is a subgraph of itself
- *Antisymmetry*: if $G_1$ is a subgraph of $G_2$, $G_2$ is a subgraph of $G_1$, then $G_1$ and $G_2$ are isomorphic
- *Transitivity*: if $G_1$ is a subgraph of $G_2$, $G_2$ is a subgraph of $G_3$, then $G_1$ is a subgraph of $G_3$

### Order Embeddings

Since the prediction is made based only on node embeddings, it is essential that node embeddings reflect subgraph relations. The order in the paper is defined as follow

$$
\mathbf{z}_v[i] \leq \mathbf{z}_u[i], \forall^D_{i = 1}, \text{ if } G_v \text{ is a subgraph of } G_u
$$

less-than-or-equal relation is a partial order, so it is able to reflect subgraph relations in the embedding space. 

### Subgraph Prediction Function $f(\mathbf{z}_u, \mathbf{z}_v)$

Define the function $E(\mathbf{z}_u, \mathbf{z}_v)$ as follow:

$$
E(\mathbf{z}_u, \mathbf{z}_v) = ||\max\{0, \mathbf{z}_q - \mathbf{z}_v\}||^2_2
$$

When the subgraph constraint is violated in any dimension $i$, i.e. $\mathbf{z}_q[i] > \mathbf{z}_q[i]$, $E(\mathbf{z}_u, \mathbf{z}_v)$ will increase. In other words, $E(\mathbf{z}_u, \mathbf{z}_v)$ represents how badly node embeddings violate the subgraph constraint. Therefore, if $G_v$ is a subgraph of $G_u$ then $E(\mathbf{z}_u, \mathbf{z}_v)$ should not be too large. Threshold $t$ for $E(\mathbf{z}_u, \mathbf{z}_v)$ is used in the prediction function as follow:

$$
f(\mathbf{z}_u, \mathbf{z}_v) = \begin{cases} 1 \text{ if } E(\mathbf{z}_u \mathbf{z}_v) < t \\ 0 \text{ otherwise} \end{cases}
$$

### Loss Function

Loss function is defined as follow

$$
L(\mathbf{z}_u, \mathbf{z}_v) =\sum_{(\mathbf{z}_u, \mathbf{z}_v) \in P} E(\mathbf{z}_u, \mathbf{z}_v) + \sum_{(\mathbf{z}_u, \mathbf{z}_v) \in N} \max\{0, \alpha - E(\mathbf{z}_u, \mathbf{z}_v)\}
$$

where $P$ is set of positive examples ($G_v$ is a subgraph of $G_u$) in the data batch and $N$ is set of negative examples

Minimizing this function is equivalent to

- Minimizing $\sum_{(\mathbf{z}_u, \mathbf{z}_v) \in P} E(\mathbf{z}_u, \mathbf{z}_v)$:
    
    For positive examples, $G_v$ is a subgraph of $G_u$, and $E(\mathbf{z}_u, \mathbf{z}_v)$ is the magnitude of subgraph constraint violation, so the lower $E(\mathbf{z}_u, \mathbf{z}_v)$ the better
    
- Minimizing $\sum_{(\mathbf{z}_u, \mathbf{z}_v) \in N} \max\{0, \alpha - E(\mathbf{z}_u, \mathbf{z}_v)\}$:
    
    For negative examples, i.e. $G_v$ is not a subgraph of $G_u$, $\mathbf{z}_u$ and $\mathbf{z}_v$ are expected to not reflect the subgraph relation, so $E(\mathbf{z}_u, \mathbf{z}_v)$ should not be too low. Thus, $E(\mathbf{z}_u, \mathbf{z}_v)$ should be at least $\alpha$, i.e. $E(\mathbf{z}_u, \mathbf{z}_v) \geq \alpha$, so that $\max\{0, \alpha - E(\mathbf{z}_u, \mathbf{z}_v)\} = 0$. The lower$\max\{0, \alpha - E(\mathbf{z}_u, \mathbf{z}_v)\}$ the better.
    

## References

Rex (Zhitao)Ying, Zhaoyu Lou, Jiaxuan You, Chengtao Wen, Arquimedes Canedo, and Jure Leskovec. Neural Subgraph Matching. *arXiv preprint arXiv:2007.03092*, 2020.
