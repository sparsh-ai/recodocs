---
sidebar_position: 1
---

# Field-aware Factorization Machines (FFM)

In the official FFM paper, it is empirically proven that for large, sparse datasets with many categorical features, FFM performs better. Conversely, for small and dense datasets or numerical datasets, FFM may not be as effective as FM. FFM is also prone to overfitting on the training dataset, hence one should use a standalone validation set and use early stopping when the loss increases.

Despite effectiveness, FM can be hindered by its modelling of all feature interactions with the same weight, as not all feature interactions are equally useful and predictive. For example, the interactions with useless features may even introduce noises and adversely degrade the performance.

|  | For each | Learn |
| --- | --- | --- |
| Linear | feature | a weight |
| Poly | feature pair | a weight |
| FM | feature | a latent vector |
| FFM | feature | multiple latent vectors |

Field-aware factorization machine (FFM) is an extension to FM. It was originally introduced in [2]. The advantage of FFM over FM is that it uses different factorized latent factors for different groups of features. The "group" is called "field" in the context of FFM. Putting features into fields resolves the issue that the latent factors shared by features that intuitively represent different categories of information may not well generalize the correlation.

Assume we have the following dataset where we want to predict Clicked outcome using Publisher, Advertiser, and Gender:

| Dataset | Clicked | Publisher | Advertiser | Gender |
| --- | --- | --- | --- | --- |
| Train | Yes | ESPN | Nike | Male |
| Train | No | NBC | Adidas | Male |

The model $w$ for logistic regression, Poly2, FM, and FFM, is obtained by solving the following optimization problem:

$$
\underset{\pmb{w}}{\min} \ \ \frac{\lambda}{2} \left\|\pmb{w} \right\|^{2} + \sum\limits_{i=1}^m log(1 + exp(-y_{i}\phi(\pmb{w}, \pmb{x_i}))
$$

where,

- dataset contains $m$ instances $(y_i,x_i)$
- $y_i$ is the label and $x_i$ is a $n$-dimensional feature vector
- λ is a regularization parameter
- $ϕ(w,x)$ is the association between $w$ and $x$

Here is the comparison of different models:

### Linear Regression

A logistic regression estimates the outcome by learning the weight for each feature.

$$
\phi(\pmb{w}, \pmb{x}) = w_0 + \sum\limits_{i=1}^n w_i x_i
$$

In our context:

$$
\begin{align*}
&\ \phi(\pmb{w}, \pmb{x}) = \\
&\ w_0 + \\
&\ w_{ESPN}x_{ESPN} + \\
&\ w_{NIKE}x_{NIKE} + \\
&\ w_{ADIDAS}x_{ADIDAS} + \\
&\ w_{NBC}x_{NBC} + \\
&\ w_{MALE}x_{MALE}
\end{align*}
$$

### Degree-2 Polynomial Mappings (Poly2)

A Poly2 model captures this pair-wise feature interaction by learning a dedicated weight for each feature pair.

$$
\phi(\pmb{w}, \pmb{x}) = w_0 + \sum\limits_{i=1}^n w_i x_i + \sum\limits_{i=1}^n \sum\limits_{j=i + 1}^n w_{h(i, j)} x_i x_j
$$

In our context:

$$
\begin{align*}
&\ \phi(\pmb{w}, \pmb{x}) = \\
&\ w_0 + \\
&\ w_{ESPN}x_{ESPN} + \\
&\ w_{NIKE}x_{NIKE} + \\
&\ w_{ADIDAS}x_{ADIDAS} + \\
&\ w_{NBC}x_{NBC} + \\
&\ w_{MALE}x_{MALE} + \\
&\ w_{ESPN, NIKE}x_{ESPN}x_{NIKE} + \\
&\ w_{ESPN, MALE}x_{ESPN}x_{MALE} + \\
&\ ...
\end{align*}
$$

![*Poly2 model - A dedicated weight is learned for each feature pair (linear terms ignored in diagram).*](/img/docs/ffm_poly2.png)

*Poly2 model - A dedicated weight is learned for each feature pair (linear terms ignored in diagram).*

However, a Poly2 model is computationally expensive as it requires the computation of all feature pair combinations. Also, when data is sparse, there might be some unseen pairs in the test set.

### Factorization Machines

FM solves this problem by learning the pairwise feature interactions in a latent space. Each feature has an associated latent vector. The interaction between two features is an inner-product of their respective latent vectors.

$$
\phi(\pmb{w}, \pmb{x}) = \textit{w}_{0} + \sum\limits_{i=1}^n w_i x_i + \sum\limits_{i=1}^n \sum\limits_{j=i + 1}^n \langle \mathbf{v}_{i} \cdot \mathbf{v}_{j} \rangle x_i x_j
$$

In our context:

$$
\begin{align*}
&\ \phi(\pmb{w}, \pmb{x}) = \\
&\ w_0 + \\
&\ w_{ESPN}x_{ESPN} + \\
&\ w_{NIKE}x_{NIKE} + \\
&\ w_{ADIDAS}x_{ADIDAS} + \\
&\ w_{NBC}x_{NBC} + \\
&\ w_{MALE}x_{MALE} +  \\
&\ \langle \textbf{v}_{ESPN, k} \cdot \textbf{v}_{NIKE, k} \rangle x_{ESPN} x_{NIKE} + \\
&\ \langle \textbf{v}_{ESPN, k} \cdot \textbf{v}_{MALE, k} \rangle x_{ESPN} x_{MALE} + \\
&\ \langle \textbf{v}_{NIKE, k} \cdot \textbf{v}_{MALE, k} \rangle x_{NIKE} x_{MALE} + \\
&\ ...
\end{align*}
$$

![*Factorization Machines - Each feature has one latent vector, which is used to interact with any other latent vectors (linear terms ignored in diagram).*](/img/docs/factorization_machine.png)

*Factorization Machines - Each feature has one latent vector, which is used to interact with any other latent vectors (linear terms ignored in diagram).*

### Field-aware Factorization Machines

FFM addresses this issue by splitting the original latent space into smaller latent spaces specific to the fields of the features.

$$
\phi(\pmb{w}, \pmb{x}) = w_0 + \sum\limits_{i=1}^n w_i x_i + \sum\limits_{i=1}^n \sum\limits_{j=i + 1}^n \langle \mathbf{v}_{i, f_{2}} \cdot \mathbf{v}_{j, f_{1}} \rangle x_i x_j
$$

![Untitled](/img/docs/ffm_equation.png)

![*Field-aware Factorization Machines - Each feature has several latent vectors, one of them is used depending on the field of the other feature (linear terms ignored in diagram).*](/img/docs/ffm_example.png)

*Field-aware Factorization Machines - Each feature has several latent vectors, one of them is used depending on the field of the other feature (linear terms ignored in diagram).*

## Links

1. [https://wngaw.github.io/field-aware-factorization-machines-with-xlearn](https://wngaw.github.io/field-aware-factorization-machines-with-xlearn/)