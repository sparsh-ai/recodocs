# GAT

GAT stands for Graph Attention Networks. This is a special GNN model that addresses several key challenges of spectral models, such as poor ability of generalization from a specific graph structure to another and sophisticated computation of matrix inverse. GAT utilizes attention mechanisms to aggregate neighborhood features (embeddings) by specifying different weights to different nodes.

![Untitled](/img/content-models-raw-mp2-gat-untitled.png)

:::info research paper

[Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio, “*Graph Attention Networks*”. ICLR, 2018.](https://arxiv.org/abs/1710.10903v3)

> We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods' features, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront. In this way, we address several key challenges of spectral-based graph neural networks simultaneously, and make our model readily applicable to inductive as well as transductive problems. Our GAT models have achieved or matched state-of-the-art results across four established transductive and inductive graph benchmarks: the Cora, Citeseer and Pubmed citation network datasets, as well as a protein-protein interaction dataset (wherein test graphs remain unseen during training).
> 

:::

<iframe width="727" height="409" src="[https://www.youtube.com/embed/uFLeKkXWq2c](https://www.youtube.com/embed/uFLeKkXWq2c)" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

GATs work on graph data. A graph consists of nodes and edges connecting nodes. For example, in Cora dataset the nodes are research papers and the edges are citations that connect the papers.

GAT uses masked self-attention, kind of similar to [transformers](https://nn.labml.ai/transformers/mha.html). GAT consists of graph attention layers stacked on top of each other. Each graph attention layer gets node embeddings as inputs and outputs transformed embeddings. The node embeddings pay attention to the embeddings of other nodes it’s connected to.

The idea is to compute the hidden representations of each node in the graph, by attending over its neighbors, following a self-attention strategy. The attention architecture has several interesting properties: (1) the operation is efficient, since it is parallelizable across node neighbor pairs; (2) it can be applied to graph nodes having different degrees by specifying arbitrary weights to the neighbors; and (3) the model is directly applicable to inductive learning problems, including tasks where the model has to generalize to completely unseen graphs.

![An illustration of multi-head attention (with K = 3 heads) by node 1 on its neighborhood. Different arrow styles and colors denote independent attention computations. The aggregated features from each head are concatenated or averaged to obtain $\vec{h}'_1$.](/img/content-models-raw-mp2-gat-untitled-1.png)

An illustration of multi-head attention (with K = 3 heads) by node 1 on its neighborhood. Different arrow styles and colors denote independent attention computations. The aggregated features from each head are concatenated or averaged to obtain $\vec{h}'_1$.

Formally,

$$
\vec{h'}_i = \sigma \left( \sum_{j \in \mathcal{N_i}} \alpha_{ij}W\vec{h}_j \right)
$$

To stabilize the learning process of self-attention, we have found extending our mechanism to employ multi-head attention to be beneficial, similarly to Vaswani et al. (2017). Specifically, K independent attention mechanisms execute the transformation, and then their features are concatenated, resulting in the following output feature representation:

$$
\vec{h'}_i = \sigma \left( \frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N_i}} \alpha_{ij}^k W^k \vec{h}_j \right)
$$

If we perform multi-head attention on the final (prediction) layer of the network, concatenation is no longer sensible—instead, we employ averaging, and delay applying the final nonlinearity (usually a softmax or logistic sigmoid for classification problems) until then.

<p><center><figure><img src='https://github.com/recohut/graph-embeddings/raw/3ae14e9b7e26389dede9d33d96465a674b8acd21/docs/_images/C047239_2.png'><figcaption>A t-SNE plot of the computed feature representations of a pre-trained GAT model’s first hidden layer on the Cora dataset. Node colors denote classes. Edge thickness indicates aggregated normalized attention coefficients between nodes i and j, across all eight attention heads (\sum_{k=1}^K \alpha_{ij}^k + \alpha{ji}^k).</figcaption></figure></center></p>

## Links

- [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)
- [https://arxiv.org/abs/1710.10903v3](https://arxiv.org/abs/1710.10903v3)
- [https://nn.labml.ai/graphs/gat/index.html](https://nn.labml.ai/graphs/gat/index.html)
