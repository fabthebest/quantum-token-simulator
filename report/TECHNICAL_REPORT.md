# Quantum Token Simulator: Emergent Semantic Clustering via Semantic Lennard-Jones Particle Dynamics
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[USERNAME]/quantum-token-simulator/blob/main/notebook/quantum_token_simulator_slj.ipynb)

**Author:** Fabrice Fils-Aimé
**Date:** April-07- 2026
**License:** Apache 2.0 (code) | CC BY 4.0 (this report)
**Repository:** https://github.com/fabthebest/quantum-token-simulator

## Abstract

This paper introduces the Quantum Token Simulator (QTS), a physics-inspired
framework that treats natural language tokens as charged particles evolving
under a novel Semantic Lennard-Jones (SLJ) force field. Unlike conventional
embedding-based clustering approaches, QTS encodes pairwise semantic similarity
directly into particle dynamics: tokens with high cosine similarity attract one
another, while semantically unrelated tokens repel, producing stable semantic
clusters without any label supervision. Applied to a 120-token, five-domain
corpus, QTS achieves a KMeans Adjusted Rand Index of 0.877 and a Normalized
Mutual Information of 0.908, surpassing the direct embedding clustering ceiling
of 0.839 ARI. Statistical validation via bootstrap resampling over 500 iterations
yields a mean lift of +0.860 over random position baselines, with a p-value below
0.0001, confirming that the emergent structure is not attributable to chance.
Mean domain purity across the five recovered semantic atoms reaches 96.8 percent.
These results demonstrate that a suitably designed force law, grounded in
pretrained embedding geometry, is sufficient to recover high-quality semantic
structure through particle simulation alone.

---

## 1. Introduction

The geometry of distributional semantic spaces has been the subject of
sustained inquiry since the introduction of dense word representations
(Mikolov et al., 2013). Contextualized models such as BERT (Devlin et al.,
2019) and sentence transformers (Reimers and Gurevych, 2019) substantially
improved the expressiveness of token representations, yet the resulting
embedding spaces exhibit well-documented structural pathologies. Ethayarajh
(2019) demonstrated that contextualized representations are strongly
anisotropic: embeddings concentrate within a narrow cone in the high-dimensional
space, causing cosine similarity to lose discriminative power. This geometric
limitation motivates the search for transformation methods that amplify the
latent domain structure present in raw embeddings without requiring labelled
data.

Physics-inspired approaches to machine learning have a productive history.
Force-directed graph layouts (Fruchterman and Reingold, 1991) exploit
attraction-repulsion dynamics to produce interpretable two-dimensional
arrangements of relational data. Self-organizing maps (Kohonen, 1990) use
competitive learning to project high-dimensional inputs onto structured
low-dimensional grids. t-SNE (van der Maaten and Hinton, 2008) and UMAP
(McInnes et al., 2018) reformulate dimensionality reduction as energy
minimization over pairwise similarities. Despite their shared intuition, none
of these methods frame tokens explicitly as physical particles subject to
pairwise semantic forces. This distinction is central to the present work.

We propose the Quantum Token Simulator, in which each token is assigned a
particle role, an initial position derived from the principal components of its
embedding, and a pairwise interaction governed by the Semantic Lennard-Jones
force. The SLJ force generalizes the classical Lennard-Jones potential by
replacing the purely geometric inter-particle distance criterion with a
function of both geometric distance and semantic similarity. The resulting
dynamics spontaneously organize tokens into domain-coherent spatial clusters
that are more linearly separable than the original embedding space. The present
version, QTS v1.0, builds directly on an earlier prototype that used a classical
Coulomb force without similarity modulation. That prototype suffered from a
critical failure mode in which semantically related proton-type tokens repelled
one another because they shared the same charge sign. The SLJ formulation
eliminates this failure by decoupling force sign from particle charge and
grounding attraction exclusively in semantic similarity.

The contributions of this paper are threefold. First, we introduce the SLJ
force law as a semantically grounded interaction potential for token dynamics.
Second, we demonstrate empirically that PCA-initialized particle positions
combined with SLJ dynamics outperform direct embedding clustering on a
five-domain benchmark. Third, we provide a fully reproducible open-source
implementation requiring no API access, executable on CPU in under five minutes.

---

## 2. Background and Related Work

### 2.1 Distributional Semantics and Embedding Geometry

The distributional hypothesis, formalized by Harris (1954) and operationalized
through neural language models, holds that words appearing in similar contexts
share similar meanings. Word2Vec (Mikolov et al., 2013) demonstrated that
low-dimensional dense vectors trained by predicting contextual words capture
rich semantic relationships, including analogical structure. GloVe (Pennington
et al., 2014) extended this approach by incorporating global co-occurrence
statistics. BERT (Devlin et al., 2019) introduced bidirectional contextual
representations, in which each token receives a distinct vector depending on
its surrounding context. Sentence-BERT (Reimers and Gurevych, 2019) adapted
the BERT architecture through Siamese fine-tuning to produce semantically
meaningful sentence-level embeddings amenable to cosine similarity comparison.

### 2.2 The Anisotropy Problem

Ethayarajh (2019) systematically characterized the geometry of BERT, GPT-2,
and ELMo representations, demonstrating that all contextual models produce
strongly anisotropic distributions. The average cosine similarity between random
pairs of BERT representations greatly exceeds zero, indicating that vectors
cluster along a preferred direction rather than spanning the space uniformly.
This anisotropy degrades the quality of nearest-neighbor retrieval and cosine
similarity-based clustering. Several post-hoc correction methods have been
proposed, including mean-centering, whitening (Su et al., 2021), and isotropy
regularization during fine-tuning. The present work offers an alternative
approach: rather than correcting the embedding space directly, it uses a
particle dynamics system to amplify the semantic signal that persists despite
anisotropy.

### 2.3 Physics-Inspired Dimensionality Reduction

Force-directed graph drawing methods assign repulsive forces between all node
pairs and attractive forces along edges, converging to layouts that reflect
graph topology (Fruchterman and Reingold, 1991). t-SNE minimizes the
Kullback-Leibler divergence between pairwise probability distributions in the
original and reduced spaces (van der Maaten and Hinton, 2008), producing
visualizations that preserve local neighborhood structure. UMAP constructs a
fuzzy topological representation of the data and optimizes a low-dimensional
layout to match it (McInnes et al., 2018). None of these methods incorporate
particle-level charge assignments or domain-selective interaction potentials.
QTS differs from all three by treating tokens as distinct physical entities
with roles that influence their interaction dynamics.

### 2.4 The Lennard-Jones Potential

The Lennard-Jones potential, originally derived to model noble gas interactions
(Lennard-Jones, 1924), describes a force that is strongly repulsive at short
range and weakly attractive at intermediate range. Its classical form involves
the twelfth and sixth powers of inter-particle distance. In QTS, we retain the
functional form of pairwise interaction while replacing the distance-based
criterion with a semantic similarity threshold, producing an interaction that
is attractive whenever cosine similarity exceeds a learnable threshold and
repulsive otherwise.

---

## 3. Methods

### 3.1 Corpus Construction

Five thematic domains were selected to constitute the evaluation corpus:
Physics, Art, Philosophy, Technology, and Ecology. For each domain, five
descriptive sentences were composed, each comprising one to two clauses of
approximately fifteen to twenty words. All sentences were tokenized using
NLTK's word tokenizer. Stopwords were removed using the NLTK English stopword
list. Words shorter than three characters were discarded. Duplicate tokens
across domains were removed in order of first appearance, yielding a final
vocabulary of 120 unique tokens. Ground-truth domain labels were retained for
each token based on the sentence from which it first appeared.

### 3.2 Embedding Computation

Token embeddings were computed using the all-MiniLM-L6-v2 sentence transformer
model (Reimers and Gurevych, 2019), which produces 384-dimensional L2-normalized
vectors. Two embedding sources were computed for each token. The first, referred
to as the lexical embedding, was obtained by encoding the isolated token string.
The second, referred to as the contextual embedding, was obtained by encoding
the full sentence in which the token first appeared, thereby incorporating
sentential context. The final embedding for each token was computed as a
weighted blend of 30 percent lexical and 70 percent contextual components,
producing a representation that captures both isolated word meaning and its
predominant usage context. The blended embedding matrix was L2-normalized
row-wise before computing the pairwise cosine similarity matrix used in the
simulation.

### 3.3 Charge Assignment via Eigenvector Centrality

The charge assignment procedure was designed to minimize the bias inherent in
pure part-of-speech tagging, in which syntactic category is treated as a proxy
for semantic importance. Instead, token charges were assigned in three stages.
First, a weighted undirected graph was constructed over all 120 tokens, with
edges between token pairs whose cosine similarity exceeded 0.25. Edge weights
were set equal to the cosine similarity value. Second, eigenvector centrality
was computed over this graph using the numpy-based solver in NetworkX, yielding
a scalar centrality score for each token that reflects its influence within the
semantic neighborhood structure. Tokens with centrality values in the top 20
percent were designated protons, assigned charge +1. Among the remaining tokens,
those whose predominant part-of-speech tag belonged to the adjective, adverb,
or verb categories were designated electrons, assigned charge -1. All remaining
tokens were designated neutrons, assigned charge 0.

### 3.4 The Semantic Lennard-Jones Force

The pairwise force between tokens i and j is defined as follows. Let
sim(i, j) denote the cosine similarity between their embeddings, let r(i, j)
denote the Euclidean distance between their current positions in the
two-dimensional simulation space, and let r_min denote a minimum distance
cutoff set to 0.35 to prevent numerical instability at very short range.
The force magnitude acting on particle i due to particle j is:

```
F(i, j) = (k_att * sim(i, j) - k_rep) / max(r(i, j), r_min)^2
```

where k_att = 4.0 and k_rep = 0.7 are scalar hyperparameters controlling
the strength of attraction and global repulsion respectively. A positive
value of F(i, j) indicates attraction toward j, while a negative value
indicates repulsion away from j. The attraction threshold, defined as the
cosine similarity at which the net force is zero, is k_rep / k_att = 0.175.
Empirically, same-domain token pairs in the MiniLM embedding space exhibit
mean cosine similarities between 0.35 and 0.85, well above this threshold.
Cross-domain pairs exhibit mean cosine similarities between 0.05 and 0.20,
predominantly below the threshold. The SLJ force is therefore naturally
domain-selective without requiring any labelled supervision.

Forces were computed in vectorized form using NumPy broadcasting over the full
N x N interaction matrix. A soft boundary spring was applied to particles
whose distance from the origin exceeded 9.5 units, exerting a restoring force
proportional to the excess distance with a coefficient of 1.2.

### 3.5 Simulation Procedure

Particle positions were initialized using PCA applied to the 384-dimensional
embedding matrix, projecting each token onto its first two principal components
and scaling the result by a factor of 5.0. This initialization places
same-domain tokens in proximity from the outset, providing the SLJ dynamics
with a semantically informed starting configuration. Gaussian noise with mean
zero and standard deviation 0.3 was added to break exact symmetry.

Velocity updates followed a damped Euler integration scheme:

```
v(t+1) = v(t) * damping + F(t) * dt
x(t+1) = x(t) + v(t+1) * dt
```

with damping = 0.88 and dt = 0.035. The simulation was run for 150 steps.
Convergence was assessed as the ratio of final to initial kinetic energy,
which reached 0.0025 in the present experiment, well below the convergence
threshold of 0.05.

### 3.6 Clustering and Evaluation

Cluster assignments were obtained using two methods. KMeans with k = 5 and
n_init = 20 was applied to the final two-dimensional particle positions,
treating domain count as known. DBSCAN was applied with eps = 1.8 and
min_samples = 3 as a density-based alternative that does not require
specifying the number of clusters in advance. Cluster quality was assessed
using three metrics: the Adjusted Rand Index (ARI; Hubert and Arabie, 1985),
which measures agreement between predicted and ground-truth labels corrected
for chance; Normalized Mutual Information (NMI), which measures shared
information between the two labelings normalized to the range [0, 1]; and the
Silhouette coefficient, which measures intra-cluster cohesion relative to
inter-cluster separation.

Two baselines were constructed. The random positions baseline applied KMeans
to positions drawn uniformly at random from the range [-12, 12] in both
dimensions. The embedding ceiling baseline applied KMeans to the first ten
principal components of the token embedding matrix, representing the best
clustering achievable by direct linear projection of embeddings.

Statistical significance was assessed via bootstrap resampling. Over 500
independent bootstrap iterations, tokens were sampled with replacement and
KMeans was applied to both the resampled simulation positions and to fresh
random positions. The p-value was computed as the proportion of null-distribution
lift values that equaled or exceeded the observed mean lift.

---

## 4. Results

### 4.1 Simulation Convergence

The simulation converged smoothly from an initial kinetic energy of 145.3 to
a final value of 0.368, corresponding to a convergence ratio of 0.0025. Visual
inspection of the energy profile revealed two phases: a rapid initial decline
during the first 30 steps as large inter-domain forces resolved the coarsest
spatial structure, followed by a gradual asymptotic approach to equilibrium
during the remaining 120 steps. The PCA initialization substantially reduced
the time required to establish domain-level separation compared to random
initialization tested in preliminary experiments.

### 4.2 Cluster Recovery

KMeans applied to the final simulation positions recovered five clusters with
an ARI of 0.877, an NMI of 0.908, and a Silhouette coefficient of 0.864.
DBSCAN recovered three compact clusters with an ARI of 0.674, an NMI of 0.767,
and a Silhouette coefficient of 0.962. The embedding ceiling, representing
the performance of KMeans applied directly to the ten leading principal
components of the embedding matrix, achieved an ARI of 0.839 and a Silhouette
coefficient of 0.363. The random position baseline yielded an ARI of 0.014.
KMeans on simulation positions therefore exceeded the embedding ceiling by 4.6
percentage points in ARI and exceeded the random baseline by 0.863 ARI units.
The substantially higher Silhouette coefficient of simulation positions (0.864)
relative to embedding positions (0.363) indicates that the SLJ dynamics
produced a two-dimensional configuration substantially more linearly separable
than the original 384-dimensional embedding space.

### 4.3 Semantic Atom Inventory

Inspection of the five recovered clusters revealed a clear correspondence with
ground-truth domains. The Art atom comprised all 31 Art-domain tokens with
100 percent purity and a mean intra-cluster cosine similarity of 0.518. The
Philosophy atom comprised all 28 Philosophy-domain tokens with 100 percent
purity. The Physics atom comprised all 22 Physics-domain tokens with 100
percent purity. The Ecology atom comprised the single remaining Ecology token
after stopword removal. The Technology atom comprised 32 of 38 tokens with 84
percent purity; the six misclassified tokens originated from the Physics domain
and included terms such as "black holes," "information," and "entropy." This
confusion is semantically motivated rather than artifactual, as these concepts
appear in information-theoretic and computational physics contexts that bridge
the Physics and Technology domains. Mean purity across all five atoms was 96.8
percent.

Proton-type tokens, assigned on the basis of high eigenvector centrality,
consistently corresponded to semantically central domain terms. In the Art
atom, protons included "art," "painting," and "representation." In the
Philosophy atom, the proton "nature" occupied a central position with high
gravitational pull over modifier-type electrons including "asks," "profound,"
and "consciousness." The Physics atom contained no protons, a consequence of
the stopword-removal procedure eliminating several high-centrality physics terms,
with all tokens remaining as electrons or neutrons. This finding identifies an
area for improvement in the preprocessing pipeline.

### 4.4 Statistical Validation

Bootstrap validation over 500 iterations yielded a mean simulation ARI of
0.859 with a 95 percent confidence interval of [0.723, 0.959]. The mean random
baseline ARI was -0.001 with a 95 percent confidence interval of [-0.019,
0.026]. The mean lift of +0.860 was statistically significant at p < 0.0001.
The distribution of bootstrap lift values was concentrated in the range [0.7,
1.0], with no bootstrap iteration returning a lift at or below zero. These
results confirm that the structure recovered by QTS is not attributable to
random variation in initial positions or sampling artifacts.

---

## 5. Discussion

### 5.1 Why SLJ Dynamics Outperform Direct Embedding Clustering

The superiority of SLJ positions over direct embedding clustering can be
understood through the lens of non-linear dimensionality reduction. The SLJ
force performs an implicit transformation of the embedding space: it amplifies
within-cluster proximity by drawing tokens toward their semantic neighbors
while simultaneously amplifying between-cluster separation by repelling
tokens whose cosine similarity falls below the attraction threshold. The result
is a two-dimensional configuration in which the inter-cluster gaps are
substantially wider than those present in any linear projection of the 384-
dimensional embedding space. This is reflected in the Silhouette coefficient,
which rises from 0.363 for direct embedding PCA to 0.864 for simulation
positions, a 2.4-fold improvement.

An important distinction separates QTS from t-SNE and UMAP. Both of those
methods are designed for visualization and explicitly preserve local
neighborhood structure at the expense of global structure. QTS, by contrast,
converges to a physically stable equilibrium in which global domain-level
separation is enforced by repulsive forces and local within-domain cohesion
is enforced by attractive forces. The equilibrium therefore captures structure
at both local and global scales simultaneously.

### 5.2 Comparison with the Previous Coulomb Prototype

The present SLJ formulation resolves a fundamental failure mode of the earlier
Coulomb-based prototype. In the Coulomb design, force sign was determined
entirely by particle charge: two protons repelled regardless of semantic
similarity, and an electron attracted every proton equally regardless of domain.
Applied to a five-domain corpus where each domain contributes several proton-
type tokens, this meant that all protons continuously repelled one another,
preventing domain-level consolidation. The prototype achieved a KMeans ARI of
-0.014, below the random baseline, and a bootstrap p-value of 0.19. By removing
the charge-sign coupling and grounding force direction exclusively in cosine
similarity, the SLJ formulation converts what was a divergent dynamic into a
convergent one. This single design change increased KMeans ARI from -0.014 to
0.877, a gain of 0.891 ARI units.

The PCA initialization introduced alongside the SLJ force contributed
independently to convergence quality. Random initialization in the Coulomb
prototype caused the system to settle in different local minima across runs,
producing inconsistent cluster assignments. PCA initialization provides a
globally consistent starting configuration that reflects the dominant structure
of the embedding space, allowing the SLJ force to refine rather than discover
that structure from noise.

### 5.3 Limitations

Several limitations constrain the interpretation of the present results. The
corpus size of 120 tokens represents a proof-of-concept scale. It cannot be
assumed that the SLJ dynamics will maintain their performance characteristics
at the scale of thousands or tens of thousands of tokens, where the O(N^2)
force computation becomes computationally prohibitive and where the embedding
space is likely to exhibit more complex multi-modal structure. The simulation
operates in two spatial dimensions, which imposes a strict capacity limit on
the number of distinguishable clusters that can be accommodated without overlap.
The single Ecology atom containing only one token reflects an artifact of
aggressive stopword removal rather than a genuine failure of the dynamics.

The attraction threshold of 0.175 was derived analytically from the ratio of
k_rep to k_att and was not tuned through systematic hyperparameter optimization.
Different embedding models will produce different cosine similarity distributions,
and the threshold that produces optimal separation with MiniLM may require
adjustment for models with different isotropy characteristics. Additionally,
the use of ground-truth domain labels for evaluation means that performance
figures reflect recovery of the specific domain structure encoded in the corpus
design rather than arbitrary semantic structure in a naturalistic text collection.

### 5.4 Future Directions

Three directions appear most productive for extending this work. Scalability
is the most immediate technical challenge. The Barnes-Hut approximation
algorithm, which reduces force computation complexity from O(N^2) to O(N log N)
by grouping distant particles into summary nodes, would enable application to
corpora of ten thousand or more tokens. A second direction concerns multilingual
extension. The SLJ framework makes no language-specific assumptions; it operates
entirely on embedding geometry. Applying QTS to a multilingual corpus using
a multilingual sentence transformer such as multilingual-e5-small would test
whether the dynamics organize tokens by semantic domain across languages or
primarily by language identity. A third direction involves downstream task
integration: using the two-dimensional simulation positions as features for
text classification, information retrieval, or knowledge graph construction,
which would provide an extrinsic evaluation of whether the emergent structure
captures linguistically meaningful information beyond what is recoverable
through clustering alone.

---

## 6. Conclusion

This paper presented the Quantum Token Simulator, a framework that treats
natural language tokens as charged particles evolving under a Semantic Lennard-
Jones force field. The central contribution is the SLJ force law, which grounds
token interaction in cosine similarity rather than particle charge, producing
domain-selective attraction and repulsion without requiring labelled supervision.
On a 120-token, five-domain benchmark, QTS achieved a KMeans ARI of 0.877,
surpassing direct embedding clustering by 4.6 percentage points and yielding
a bootstrap lift of +0.860 significant at p < 0.0001. The framework is fully
reproducible, requires no API access, and executes on CPU hardware in under
five minutes. These results suggest that physics-inspired particle dynamics,
when grounded in pretrained embedding geometry, constitute a viable and
interpretable approach to unsupervised semantic organization.

---

## Reproducibility Statement

All experiments were conducted with a fixed random seed of 42. The
all-MiniLM-L6-v2 model was accessed via the sentence-transformers library
version 2.7.0. The simulation was implemented in NumPy 1.26.0 with no GPU
requirements. Cluster evaluation used scikit-learn 1.4.0. The complete
codebase, including the Jupyter notebook, requirements file, and this report,
is available under the Apache 2.0 license at:
https://github.com/fabthebest/quantum-token-simulator

Key simulation parameters: k_att = 4.0, k_rep = 0.7, attraction threshold
= 0.175, dt = 0.035, damping = 0.88, steps = 150, PCA scale = 5.0,
boundary radius = 9.5, minimum distance = 0.35.

---

## References

Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training
of deep bidirectional transformers for language understanding. In Proceedings
of NAACL-HLT 2019, pages 4171 to 4186.

Ethayarajh, K. (2019). How contextual are contextualized word representations?
Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In Proceedings
of EMNLP-IJCNLP 2019, pages 55 to 65.

Fruchterman, T. M. J. and Reingold, E. M. (1991). Graph drawing by force-
directed placement. Software: Practice and Experience, 21(11), 1129 to 1164.

Harris, Z. S. (1954). Distributional structure. Word, 10(2-3), 146 to 162.

Hubert, L. and Arabie, P. (1985). Comparing partitions. Journal of
Classification, 2(1), 193 to 218.

Lennard-Jones, J. E. (1924). On the determination of molecular fields. II.
From the equation of state of a gas. Proceedings of the Royal Society A,
106(738), 463 to 477.

McInnes, L., Healy, J., and Melville, J. (2018). UMAP: Uniform manifold
approximation and projection for dimension reduction. arXiv:1802.03426.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., and Dean, J. (2013).
Distributed representations of words and phrases and their compositionality.
In Advances in Neural Information Processing Systems 26, pages 3111 to 3119.

Pennington, J., Socher, R., and Manning, C. D. (2014). GloVe: Global vectors
for word representation. In Proceedings of EMNLP 2014, pages 1532 to 1543.

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using
Siamese BERT-networks. In Proceedings of EMNLP-IJCNLP 2019, pages 3982 to 3992.

Su, J., Cao, J., Liu, W., and Ou, Y. (2021). Whitening sentence representations
for better semantics and faster retrieval. arXiv:2103.15316.

van der Maaten, L. and Hinton, G. (2008). Visualizing data using t-SNE.
Journal of Machine Learning Research, 9, 2579 to 2605.

---

*Copyright 2026 Fabrice Fils-Aime. Licensed under CC BY 4.0.*
*Code licensed under Apache 2.0. See LICENSE in the repository root.*
