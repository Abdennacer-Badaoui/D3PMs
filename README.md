# Structured Denoising Diffusion Models in Discrete State-Spaces
The paper "Structured Denoising Diffusion Models in Discrete State-Spaces" introduces Discrete Denoising Diffusion Probabilistic Models (D3PMs), a generative model for discrete data that builds upon the success of Denoising Diffusion Probabilistic Models (DDPMs) in continuous spaces.

Structured Denoising Diffusion Models in Discrete State-Spaces by Austin et al.: https://arxiv.org/pdf/2107.03006
## Diffusion Models for Discrete State Spaces
Diffusion models typically operate on continuous data like image pixels. This paper introduces a new approach called Discrete Denoising Diffusion Probabilistic Models (D3PMs) for data with a finite set of values, such as images with discrete color levels or text with a specific alphabet. D3PMs achieve this by employing a series of diffusion steps. In each step, a transition matrix, which captures the probability of transitioning from one state (e.g., a specific pixel value) to another, is used to gradually corrupt the data with noise. The model then learns to reverse this process, effectively denoising the data and generating new samples from the underlying distribution. This framework allows for incorporating domain knowledge by designing structured transition matrices that reflect specific data characteristics.

![Capture d’écran 2024-07-01 094153](https://github.com/Abdennacer-Badaoui/D3PMs/assets/106801897/bbbe4946-bd98-4b06-8dcc-2b3155fc3a8c)

## Categorical Distribution and Transition Matrices in D3PMs

### Categorical Distribution

Imagine a data point $x$ that can take on one of $K$ possible values ($x \in \{x_1, x_2, \ldots, x_K\}$). A categorical distribution describes the probability of $x$ belonging to each category. This is represented by a probability vector $q(x) = [q(x_1), q(x_2), \ldots, q(x_K)]$, where each element $q(x_i)$ signifies the probability of $x$ taking the value $x_i$. The sum of all probabilities in the vector equals 1:

$$
\sum_{i=1}^{K} q(x_i) = 1 
$$


### Transition Matrices

The diffusion process in D3PMs is governed by a sequence of transition matrices, one for each diffusion step $t$ (from 0 to $T-1$). Each transition matrix, denoted by $Q_t$, is a square matrix of size $K \times K$. It encodes the probability of transitioning from any state $x_i$ to any other state $x_j$ during step $t$:

$$
Q_t(x_i, x_j)
$$

Here, $Q_t(x_i, x_j)$ represents the probability of transitioning from state $x_i$ to state $x_j$ in step $t$.

$$
\sum_{j=1}^{K} Q_t(x_i, x_j) = 1, \quad \text{for all i} \in \{1, 2, \ldots, K\}
$$

These transition matrices essentially define the "corruption process" at each step. By multiplying the current data distribution $q(x_t)$ with the transition matrix $Q_t$, we obtain the distribution of the data $q(x_{t+1})$ after the noise is added in step $t$:

$$
q(x_{t+1} \mid x_t) = Q_t q(x_t)
$$

In essence, the categorical distribution captures the probabilities of the data points across different categories, while the transition matrices govern how these probabilities evolve with the noise injection during the diffusion process.

The effectiveness of D3PMs heavily relies on the concept of locality in the transition matrices. This principle dictates how the corruption process in each step is localized to the state itself or its immediate neighbors in the data space.

Here's how locality plays a role:

- **Limited Dependencies**: The transition probability $Q_t(x_i, x_j)$ should be significant only for $x_i$ and $x_j$ that are "close" to each other in the data space.
- **Reduced Complexity**: By focusing on local transitions, the model complexity and computational cost are reduced compared to matrices with global dependencies.
- **Incorporating Domain Knowledge**: Locality allows us to design transition matrices that reflect specific data characteristics.

Examples of Transition Matrices:

- **Uniform Transition Matrices**: In a uniform transition matrix, each state has an equal probability of transitioning to any other state. This means that no matter what the current state is, it is equally likely to move to any of the possible states in the next step.
- **Absorbing State Matrices**: In an absorbing state matrix, certain states are designated as absorbing states. Once the process transitions into an absorbing state, it remains there indefinitely. For example, if state $K$ is an absorbing state, once the process reaches state $K$, it will stay in state $K$ and not transition to any other state.
- **Discretized Gaussian Matrices**: In a discretized Gaussian transition matrix, the probability of transitioning from one state to another is based on a Gaussian distribution. States that are closer to each other have higher transition probabilities, while states that are further apart have lower transition probabilities. This creates a smooth transition pattern where the process is more likely to move to nearby states.


## Forward Process in D3PMs

The forward process in D3PMs refers to the gradual corruption of clean data with noise over a series of discrete steps. This process relies on categorical distributions, transition matrices, and the crucial property of convergence to a stationary distribution. Here's a breakdown with mathematical notation:

### 1. Initial State

We start with clean data $x$ represented by a categorical distribution $q(x) = [q(x_1), q(x_2), \ldots, q(x_K)]$.
Each element $q(x_i)$ signifies the probability of $x$ belonging to category $x_i$.

### 2. Diffusion Steps

The corruption unfolds over $T$ diffusion steps ($t = 0$ to $T-1$):

- In each step $t$, a transition matrix $Q_t$ (size $K \times K$) is applied.
- This matrix encodes the probability of transitioning from any state $x_i$ to any other state $x_j$ during step $t$: $Q_t(x_i, x_j)$.

#### Mathematical Notation

The core of the forward process is iteratively updating the data distribution based on the transition matrix:

$$ q(x_{t+1} \mid x_{t}) = Q_t q(x_t) $$

The resulting $q(x_{t+1})$ represents the distribution of the data after noise is added in step $t$.

### 3. Iterative Corruption and Convergence

This process is repeated for all steps, with each $Q_t$ potentially different. However, there's a crucial constraint:
- The rows of the cumulative product $M_t = Q_1 \cdot Q_2 \cdot \ldots \cdot Q_t$ (representing the overall transition across all steps up to $t$) must converge to a known stationary distribution as $t$ approaches infinity.

#### Stationary Distribution

A stationary distribution, denoted by $\pi = [\pi_1, \pi_2, \ldots, \pi_K]$, is a fixed point of the diffusion process. For a large number of corruption steps, we expect $q(x_{t}) \approx \pi$. This means:

In simpler terms, after a sufficient number of diffusion steps, the data distribution stops changing and reaches a steady state represented by the stationary distribution, that we can use to sample from in the backward process.

## Backward Process

So far we have shown how to distort the original data distribution $q(x_{0})$ into $\pi(x_{T})$ with iterative addition of tractable noise. The noise here is introduced via a Markov transition matrix. However, the key ingredient of D3 models is a learned reverse process, which attempts to iteratively undo the corruption of the forward process. The reverse process is also defined via a Markov chain:

$$ p_{\theta}(x_{t-1} \mid x_{t}) = P_{\theta} \cdot p(x_{t}) $$

Note that the reverse process is conditioned on forward-looking time steps. Here the Markov transition matrix is parameterized somehow, and could actually also depend on the conditioning variable $x_{t}$. In fact, in most published diffusion models, $P_{\theta}$ is actually a neural network with arguments $x_{t}$ and $t$.

Once the model reverse process is known, we can use it to generate samples from the data distribution $p(x_0)$ by first sampling $x_{T} \sim \pi = p(x_{T})$, then iteratively generating the sequence:

$$ x_{T} \rightarrow x_{T-1} \rightarrow \ldots \rightarrow x_{0} $$

## Use case (Diffusion on the Discrete Space of Dish Names)

In this project, we explore the application of categorical diffusion processes to the Food101 dataset, focusing on the task of food classification by conditioning on image features. Although a simple convolutional neural network (CNN) could be employed to achieve effective image classification, our aim is to delve into the emerging domain of categorical diffusion. Our discrete space consists of 101 states, corresponding to the 101 categories of food in the dataset. By iteratively denoising random category indices and conditioning on the rich semantic embeddings provided by the images, we investigate the potential of diffusion models in handling categorical data. This approach not only broadens our understanding of diffusion processes in discrete spaces but also opens up new avenues for leveraging such models in various classification tasks.

![dataset-cover](https://github.com/Abdennacer-Badaoui/D3PMs/assets/106801897/e8d608d7-8914-4b7c-994c-2f2764c56051)



