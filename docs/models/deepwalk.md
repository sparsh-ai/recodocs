# DeepWalk

[DeepWalk](https://arxiv.org/abs/1403.6652) learns representations of online social networks graphs. By performing random walks to generate sequences, the paper demonstrated that it was able to learn vector representations of nodes (e.g., profiles, content) in the graph.

![/img/content-models-raw-mp2-deepwalk-untitled.png](/img/content-models-raw-mp2-deepwalk-untitled.png)

### **The DeepWalk process operates in a few steps**

1. For each node, perform N “random steps” starting from that node
2. Treat each walk as a sequence of node-id strings
3. Given a list of these sequences, train a word2vec model using the Skip-Gram algorithm on these string sequences

### Let's see how it works in depth

- node embedding learned in an unsupervised manner

![/img/content-models-raw-mp2-deepwalk-untitled-1.png](/img/content-models-raw-mp2-deepwalk-untitled-1.png)

- highly resembles word embedding in terms of the training process
- motivation is that the distribution of both nodes in a graph and words in a corpus follow a power law
    
    ![/img/content-models-raw-mp2-deepwalk-untitled-2.png](/img/content-models-raw-mp2-deepwalk-untitled-2.png)
    
- The algorithm contains **two steps**: 1) Perform random walks on nodes in a graph to generate node sequences, 2) Run skip-gram to learn the embedding of each node based on the node sequences generated in step 1
    
    ![Different colors indicate different labels](/img/content-models-raw-mp2-deepwalk-untitled-3.png)
    
    Different colors indicate different labels
    
- However, the main issue with DeepWalk is that it lacks the ability to generalize. Whenever a new node comes in, it has to re-train the model in order to represent this node (transductive). Thus, such GNN is not suitable for dynamic graphs where the nodes in the graphs are ever-changing.

### What does it look like in code?

```python
# Instantiate a undirected Networkx graph
G = nx.Graph()
G.add_edges_from(list_of_product_copurchase_edges)

def get_random_walk(graph:nx.Graph, node:int, n_steps:int = 4)->List[str]:
   """ Given a graph and a node, 
       return a random walk starting from the node 
   """
   local_path = [str(node),]
   target_node = node
   for _ in range(n_steps):
      neighbors = list(nx.all_neighbors(graph, target_node))
      target_node = random.choice(neighbors)
      local_path.append(str(target_node))
   return local_path

walk_paths = []

for node in G.nodes():
   for _ in range(10):
      walk_paths.append(get_random_walk(G, node))
 
walk_paths[0]
>>> [‘10001’, ‘10205’, ‘11845’, ‘10205’, ‘10059’]
```

What these random walks provide to us is a series of strings that act as a path from the start node — randomly walking from one node to the next down the list. What we do next is we treat this list of strings as a sentence, then utilize these series of strings to train a Word2Vec model:

```python
# Instantiate word2vec model
embedder = Word2Vec(
   window=4, sg=1, hs=0, negative=10, alpha=0.03, min_alpha=0.0001,    
   seed=42
)

# Build Vocabulary
embedder.build_vocab(walk_paths, progress_per=2)

# Train
embedder.train(
   walk_paths, total_examples=embedder.corpus_count, epochs=20, 
   report_delay=1
)
```

It’s a slight stretch, but here’s the gist of it from the recommendations perspective:

- Use the product pairs and associated relationships to create a graph
- Generate sequences from the graph (via *random walk*)
- Learn product embeddings based on the sequences (via *word2vec*)
- Recommend products based on embedding similarity (e.g., cosine similarity, dot product)

## Links

1. [DeepWalk: Online Learning of Social Representations. arXiv, Jun 2014](https://arxiv.org/abs/1403.6652)
2. [https://github.com/phanein/deepwalk](https://github.com/phanein/deepwalk)
3. [https://towardsdatascience.com/deepwalk-its-behavior-and-how-to-implement-it-b5aac0290a15](https://towardsdatascience.com/deepwalk-its-behavior-and-how-to-implement-it-b5aac0290a15)
4. [https://towardsdatascience.com/introduction-to-graph-neural-networks-with-deepwalk-f5ac25900772](https://towardsdatascience.com/introduction-to-graph-neural-networks-with-deepwalk-f5ac25900772)
5. [https://www.analyticsvidhya.com/blog/2019/11/graph-feature-extraction-deepwalk/](https://www.analyticsvidhya.com/blog/2019/11/graph-feature-extraction-deepwalk/)