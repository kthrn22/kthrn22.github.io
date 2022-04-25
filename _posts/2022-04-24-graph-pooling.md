---
title: Graph Pooling
tags:
  - GNN
  - Pooling
---

In this blog, I present a graph $\mathcal{G} = (V, E)$, where $V$ the set of nodes and $E$ is the set of edges. I denote the adjacency matrix and node embeddings as $\mathbf{A} \in \mathbb{R}^{ \vert V \vert \times \vert V \vert}$, $\boldsymbol{Z} \in \mathbb{R}^{\vert V \vert \times d}$ respectively.

Given the node embeddings of a graph, how can we make predictions at graph-level? <br/>
$\rightarrow$ We will have to learn the embedding for the entire a graph by pooling node embeddings and this is usually referred as **graph pooling**.

In this blog, I will present some strategies for **graph pooling**

# Set Pooling Methods

The goal **Set Pooling** is to map a set of node embeddings $\{\boldsymbol{z}_1, \dots, \boldsymbol{z}_{\vert V \vert}\}$ to an embedding that represents the entire graph, $\boldsymbol{z}_{G}$.

## Global pooling

We can obtain the embedding for the graph by taking the sum, mean, or max of node embeddings. 

We can use this method for small graphs.

## Set2Set

Vinyals et al. [Order matters: Sequence to sequence for sets](https://arxiv.org/pdf/1511.06391.pdf), ICLR 2015

This method allows us to pool node embeddings using LSTMs and attention. 

Set2Set iterates for $t = 1, \dots, T$ steps <br/>

At step $t$:
* Compute the query vector for attention at iteration $t$
$$\boldsymbol{q}_t = \text{LSTM}(\boldsymbol{o}_{t - 1}, \boldsymbol{q}_{t - 1})$$
* Compute the attention score over each node using the attention function $f_a: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \R$
$$e_{v, t} = f_a(\bold z_v, \bold q_t), \forall v \in V$$ 
* Normalize the attention score to obtain the attention weights
$$\alpha_{v, t} = \frac{\exp(e_{v, t})}{\sum_{u \in V}exp(e_{u, t})}$$
* Compute the weighted sum of node embeddings, the weight for each node's embedding is its attention weight computed in the previous step
$$\bold o_t = \sum_{v \in V} \alpha_{v, t}\bold z_v, \text{ where } \bold z_v \text{ is node } v \text{'s embedding}$$ 

After $T$ iterations, compute the final embedding of the graph by the following equation:

$$ \bold z_G = \text{CONCAT}(\bold o_1, \dots, \bold o_T)$$
  
# Graph Coarsening Methods
A drawback of **Set Pooling Methods** is that they do not exploit the structure of the graph (They compute the graph embedding solely based on node embeddings). 

A strategy for taking **graph structure** into account is performing **graph coarsening** / **clustering** while pooling node representations.

Lets say we want to group nodes into $c$ clusters. We will use a clustering function $$f_c: \mathcal{G} \times \R^{|V| \times d} \rightarrow \R^{|V| \times c}$$ that maps nodes to an assignment matrix over $c$ clusters.

Suppose we have a cluster assignment matrix $$\bold S = f_c(\mathcal{G}, \bold Z) \in \R^{|V| \times c}$$ where $\bold S_{v, i}$ denotes how likely node $v$ is in cluster $i$

We will use $\bold S$ to coarsen the graph. Specifically, we will compute the coarsened adjacency matrix
$$ \bold{\hat{A}} = \bold S^T \bold A \bold S \in \R^{c\times c} $$,

and a new matrix of embeddings

$$ \bold{\hat{X}} = \bold S^T \bold Z \in \R^{c \times d}$$

> **Intuition of $\bold{\hat{A}}$ and $\bold{\hat{X}}$** <br/> 
For $i, j \leq c$
>
> $$\bold{\hat{A}}_{i, j} = \sum_{v \in V} \sum_{u \in N(v)} \bold S_{v, i} \bold S_{u,j} $$ 
> 
> Therefore, $\bold{\hat{A}}_{i, j} \in \R$ represents the weight of connections (edges) between cluster $i$ and cluster $j$. In other words, $\bold{\hat{A}}$ represents the strength of connections between each pair of $c$ clusters. ($\bold{\hat{A}}$ is the weighted adjacency matrix of the coarsened graph)
> 
> $\bold{\hat{X}}$ takes in the node embeddings, $\bold Z$, aggregrates them base on the assignment matrix $\bold S$, and outputs pooled features for clusters (the $i$-th row of $\bold{\hat{X}}$ represents the features of cluster $i$).

After grouping the nodes into $c$ clusters, we will obtain a coarsened graph that has $c$ nodes (we just reduced the number of nodes to $c$ !!), which is represented by the weighted adjacency matrix $\bold{\hat{A}}$, and new node features $\bold X$. Then we can apply GNN on this coarsened graph and repeat the whole process for $T$ times. Thus, after each iteration, the graph's size will be reduced, and we will obtain the **original graph's embedding** using node embeddings of the final coarsened graph. 

## DiffPool
Ying et al. [Hierarchical Graph Representation Learning with Differentiable Pooling](https://proceedings.neurips.cc/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf), NeurIPS 2018
<p style = "text-align: center;">
  <img src="/images/DiffPool.jpg">
  From 
  <a href = "https://proceedings.neurips.cc/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf">Source
  </a>
</p>

DiffPool stacks $L$ GNN modules and learns to assign nodes to clusters at each layer using the embeddings of the previous layer. Therefore, DiffPool learns node embeddings that benefit both graph classification and hierarchical pooling.

DiffPool consists of 2 phases: **Pooling nodes** and **Learning cluster assignment matrix**

**Pooling nodes**

Denote the learned cluster assignment matrix at layer $l$ as $\bold S^{(l)} \in \R^{n_l \times n_{l + 1}}$, where $n_l$ is the number of nodes at layer $l$ and $n_{l + 1}$ is the number of nodes at layer $l + 1$. Denote the adjacency matrix and node embedding matrix at layer $l$ as $\bold A^{(l)} \in \R^{n_l \times n_l}$ and $\bold Z^{(l)} \in \R^{n_l \times d}$ respectively. DiffPool coarsens the graph at layer $l$ by the following equations:

$$ \bold A^{(l + 1)} = \bold S^{(l)^T} \bold A^{(l)} \bold S^{(l)} \in \R^{n_{l + 1} \times n_{l + 1}}$$
$$\bold X^{(l + 1)}  = \bold S^{(l)} \bold Z^{(l)} \in \R^{n_{l + 1} \times d}$$

$\bold X^{(l + 1)}$ and $\bold A^{(l + 1)}$ represent the node features and weighted adjacency matrix of the coarsened graph, which is the graph at layer $l + 1$. These two matrices will be used at inputs to layer $l + 1$.

**Learning cluster assignment matrix**

At layer $l$, given the coarsened adjacency matrix $\bold A^{(l)}$ and node features $\bold X^{(l)}$, how can we learn the assignment matrix $\bold S^{(l)}$ and the node embeddings $\bold Z^{(l)}$?

DiffPool uses 2 seperate GNNs to learn $\bold S^{(l)}$ and $\bold Z^{(l)}$, which is $\text{GNN}^{(l)}_{pool}$ and $\text{GNN}^{(l)}_{embed}$ respectively:

$$ \bold Z^{(l)} = \text{GNN}^{(l)}_{embed} (\bold A^{(l)}, \bold X^{(l)})$$

$\text{GNN}^{(l)}_{embed}$ uses the adjacency matrix $\bold A^{(l)}$ and node features $\bold X^{(l)}$ to generate node embeddings $\bold Z^{(l)}$

$$\bold S^{(l)} = \text{softmax}(\text{GNN}^{(l)}_{embed}(\bold A^{(l)}, \bold S^{(l)})) $$

where $\text{softmax}$ is applied in the row-wise way, since $\bold S^{(l)}$ represents the probabilistic assignment for each node.  

At layer $0$, the input will be the original graph's adjacency matrix $\bold A$ and node features $\bold X$. At the second to last layer $L - 1$, we set the cluster assignment matrix $\bold S^{(L - 1)}$ to a vector of $1$'s, meaning that the nodes at layer $L - 1$ will be assigned into a single cluster, and layer $L$ will generate a final embedding vector, which represents the **original graph's embedding**. 

> **Permutation Invariance**<br/>
> In order to be used in graph classification, DiffPool should be permutation invariant.
>
> ***Proposition***. Let $P$ be any permutation matrix. Then $\text{DiffPool}(\bold A, \bold Z) = \text{DiffPool}(P\bold A P^T, P \bold Z)$ as long as $\text{GNN}(\bold A, \bold X) = \text{GNN}(P\bold A P^T, P \bold X)$. ($\text{GNN}$ refers to the GNN module used in DiffPool)
> 
> *Proof*. <br/>
> Let 
> $$ \begin{split}
(\bold S, \bold Z) & = \text{GNN}(\bold A, \bold X) \\
(\bold S_P, \bold Z_P) & = \text{GNN}(P \bold A P^T, P \bold X) \\
(\bold{\hat{A}}, \bold{\hat{X}}) & = \text{DiffPool}(\bold A, \bold Z) \\
(\bold{\hat{A}}_P, \bold{\hat{X}}_P) & = \text{DiffPool}(P \bold A P^T, P \bold Z) \\
\end{split}$$
> In order to prove the the permutation invariance of $\text{DiffPool}$, we will prove that $\bold S = \bold S_P$, $\bold Z = \bold Z_P$, $\bold{\hat{A}} = \bold{\hat{A}}_P$, and $\bold{\hat{X}} = \bold{\hat{X}}_P$
>
> Since $\text{GNN}(\bold A, \bold X) = \text{GNN}(P\bold A P^T, P \bold X)$, $\bold S = \bold S_P$ and $\bold Z = \bold Z_P$.
>
> We have
> $$ \begin{split}
\bold{\hat{A}}_P & = (P \bold S)^T(P\bold A P^T)(P\bold S) \\
& = \bold S^T(P^TP)\bold A(P^TP)\bold S \\
& = \bold S^T \bold A \bold S ~~~\text{(since } P^TP = I)\\
& = \bold{\hat{A}} \\
\\
\bold{\hat{X}}_P & = (P\bold S)^T(P \bold Z) \\
& = \bold S^T(P^TP)\bold Z \\
& = \bold S^T \bold Z ~~~\text{(since } P^TP = I)\\
& = \bold{\hat{X}}
\end{split}$$
> 
> Thus, the proposition is proved. 

**Auxiliary Link Prediction Objective and Entropy Regularization**

In pratice training the pooling GNN $(\text{GNN}_{pool})$
based only on gradient in the graph classification task can be difficult, since optimizing the $\text{GNN}_{pool}$ will now become a non-convex optimization problem. In order to address this problem, the authors of DiffPool train $\text{GNN}_{pool}$ with an auxiliary link prediction objective, which tells us that nearby nodes should be pooled together. Specifically, at each layer $l$, the following loss function will be minimized: 

$$L_{LP} = ||\bold A^{(l)} - \bold S^{(l)} \bold S^{(l)^T}||_
$$ L_{LP} =  ||\bold A^{(l)} - \bold S^{(l)} \bold S^{(l)^T} ||_F$$ 
where $||.||_F$ denotes the Forbenius norm.

> **Intuition of $L_{LP}$**
> 
> At each layer $l$, let $\bold Q = \bold A^{(l)} - \bold S^{(l)} \bold S^{(l)^T}$, then $L_{LP} = \sqrt{\sum_{i \in V} \sum_{j \in V} \bold Q_{ij}^2}$  
>
> We have $\bold Q_{ij} = \bold A^{(l)}_{ij} - \bold S^{(l)}_i \bold S^{(l)^T}_j$, where $\bold S^{(l)}_i$ is the $i$-th row of $\bold S^{(l)}$.
>
> Thus, minimizing $L_{LP}$ means minimizing $\sum_{i \in V} \sum_{j \in V} (\bold A^{(l)}_{ij} - \bold S^{(l)}_i \bold S^{(l)^T}_j)^2 $. 
>
> If $i$ and $j$ are nearby then both $\bold A^{(l)}_{ij}$ and $\bold S^{(l)}_i \bold S^{(l)^T}_j$ will be large. Similarly, if $i$ and $j$ are not nearby then both $\bold A^{(l)}_{ij}$ and $\bold S^{(l)}_i \bold S^{(l)^T}_j$ will be small. Thus minimizing $L_{LP}$ will force nearby nodes being pooled together. 

Moreover, the cluster assignment matrix $\bold S^{(l)}$ learned by the pooling GNN should have row vectors that are close to one-hot vectors, so that the node assignment can be clearly defined. Therefore, the authors of DiffPool regularize the entropy of the cluster assignment by minimizing the following equation:

$$L_E = \frac{1}{|V|} \sum_{i \in V}H(\bold S^{(l)}_i) $$
where $H$ is the entropy function.

> **Intuition of $L_E$**
>
> Minimizing $L_E$ will minimize the entropy of the cluster assignment of each node, i.e forcing $\text{GNN}_{pool}$ to be confidence about assigning each node into a cluster. 

During training, $L_{LP}$ and $L_E$ of each layer will be added to the classification loss.

# Top-$K$ methods

Unlike **Graph Coarsening Methods**, instead of clustering nodes, Top-$K$ methods rank nodes and select nodes with top $k$ score. The selected nodes then will form a new graph. 

## gPool

H. Gao, S. Ji. [Graph U-Nets](http://proceedings.mlr.press/v97/gao19a/gao19a.pdf), ICML 2019

<p style = "text-align: center;">
  <img src="/images/gPool.jpg">
  From 
  <a href = "http://proceedings.mlr.press/v97/gao19a/gao19a.pdf">Source
  </a>
</p>

gPool layer selects a new subset of nodes of the original graph to form a new and smaller graph. gPool consists of 3 phases: **Projection**, **Node Selection**, and **Gate**

Denote the number of nodes in the graph at layer $l$ as $n_l$.
Denote the graph's adjacency matrix and node features at gPool layer $l$ as $\bold A^{(l)} \in \R^{n_l \times n_l}$ and $\bold X^{(l)} \in \R^{n_l \times d}$ respectively.

**Projection**

gPool layer $l$ learns a projection vector $\bold p^{(l)} \in \R^{n_l \times 1}$ to and projects node features $\bold X^{(l)}$  to 1 dimension: 
$$ \bold y = \bold X^{(l)} \frac{\bold p^{(l)}}{|| \bold p^{(l)}||} \in \R^{n_l \times 1},$$

where $\bold y_i \in \R$ is node $i$'s score and it represents how much information of node $i$ can be retained when we project node $i$'s features onto the dimension of $\bold p^{(l)}$

**Node Selection**

Since we want to preserve as much information as possible, we select $k$ nodes with the highest score and record their indices:

$$ \text{idx} = \text{rank}(\bold y, k)$$

After selecting $k$ nodes, we obtain a new graph represented by a new adjacency matrix $\bold{\hat{A}}$ and new node features $\bold{\hat{X}}$. Specifically, based on the indices of selected nodes, gPool extracts some rows and columns of $\bold{A}$ to form $\bold{\hat{A}}$ and extracts some rows of $\bold{X}$ to form $\bold{\hat{X}}$:

$$ \begin{split}
& \bold{A}^{(l + 1)} = \bold{\hat{A}} = \bold A^{(l)}(\text{idx}, \text{idx}) \in \R^{k \times k} \\
& \bold{\hat{X}} = \bold X^{(l)}(\text{idx}, :) \in \R^{k \times d}
\end{split} $$

**Gate**

In order to control the information flow, gPool applies gate operation:

$$\bold{\hat{y}} = \sigma(\bold y(\text{idx})), \text{where } \sigma(.) \text{ is the sigmoid function} $$ 

$$ \bold{X}^{(l + 1)} = \bold{\hat{X}} \odot (\bold{\hat{y}} \bold 1_d^T), \text{where } \odot \text{ is the element-wise matrix product} $$

Applying the gate operator, gPool makes the projection vector $\bold p^{(l)}$ trainable by back-propagation.

>**Graph Connectivity Problem**
>
> Since gPool omits nodes that are not in top $k$, their related edges will also be removed. As a result, the new graph formed by selected nodes will likely have isolated nodes. 
> For example, take the graph below as the original graph
> <p style = "text-align: center;">
>  <img src="/images/output.png">
> </p>
>
> Suppose that gPool retains node $2, 4, 5$ and $6$, so node $1$, $3$ and their related edges wil be omitted.
> Then our new graph will be
> <p style = "text-align: center;">
>  <img src="/images/output1.png">
> </p>
>
> In our new graph, the nodes are isolated.
> For GNN modules that aggregrate information from neighbor nodes, this could be a problem, since isolated nodes do not have neighbors to provide them information.
> 
> The authors of gPool solve this problem by using the power of the adjacency matrix. Specifically, in the **Node Selection** phase, instead of extracting rows and columns of the adjacency matrix $\bold A$ $(\bold{\hat{A}} = \bold A(\text{idx}, \text{idx}))$, we can perform extraction from $\bold A^m$, i.e $ \bold{\hat{A}} = \bold A^m(\text{idx}, \text{idx})$. $\bold A^m$ builds connections between nodes whose distances are less than or equal to $m$, so using $\bold A^m$ can increase the connectivity of the new graph. 

# References
O. Vinyals, S. Bengio, and M. Kudlur. [Order matters: Sequence to sequence for sets](https://arxiv.org/pdf/1511.06391.pdf). In *ICLR*, 2015.

Jure Leskovec. [Stanford CS224W: Machine Learning with Graphs, Lectur 8](https://web.stanford.edu/class/cs224w/slides/08-GNN-application.pdf).

W.L. Hamilton. [Graph representation learning](https://www.morganclaypool.com/doi/10.2200/S01045ED1V01Y202009AIM046). In *Synthesis Lectures on Artifical Intelligence and Machine Learning*, 14 (3) (2020), pp. 1-159.

H. Gao, S. Ji. [Graph u-nets](http://proceedings.mlr.press/v97/gao19a/gao19a.pdf). In *ICML*, 2019. <br/>

R. Ying, J. You, C. Morris, X. Ren, W. Hamilton, and J. Leskovec. [Hierarchical
graph representation learning with differentiable pooling](https://proceedings.neurips.cc/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf). In *NeurIPS*, 2018 <br/>

D. Grattarola, D. Zambon, F.M. Bianchi, and C. Alippi. [Understanding Pooling in Graph Neural Networks](https://arxiv.org/pdf/2110.05292.pdf). *arXiv preprint arXiv:2110.05292*, 2021.
