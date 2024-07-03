# Structured Denoising Diffusion Models in Discrete State-Spaces
The paper "Structured Denoising Diffusion Models in Discrete State-Spaces" introduces Discrete Denoising Diffusion Probabilistic Models (D3PMs), a generative model for discrete data that builds upon the success of Denoising Diffusion Probabilistic Models (DDPMs) in continuous spaces.
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

