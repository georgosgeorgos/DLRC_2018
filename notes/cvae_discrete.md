

## CVAE for mimicking mixtures of Gaussians

### Model

We propose the latent a variabel model for a regression model p(y \vert x)  with a likelihood
$$
p(y \vert z, x) = \mathcal{N}(y \vert \mu(z,x), \sigma(z,x)^2)
$$
And a concrete prior p(z \vert x) that tries to mimic a discrete distribution. The former is given by
$$
\begin{align}
z_k &= \frac{\exp\{\tau^{-z}(\log\pi_k(x) + \epsilon_k)\}}{\sum_k\exp\{\tau^{-z}(\log\pi_k(x) + \epsilon_k)\}}
\\
\epsilon_k&= \sim -\log{-\log(u_k)}
\\
u_k & \sim \text{Uniform(0,1)}
\end{align}
$$
And p(z \vert x) has the form 
$$
\begin{align}
	p(z \vert x) = (K-1)!{\tau}^{K-1} 
    \frac{\prod_k \pi_k(x)z_k^{-\tau-1}}{[\sum_k \pi_k(x)z_k^{-\tau}]^K}
\end{align}
$$

### Relaxing the likelihoods

Here, we will have soft z-s meaning that it will never be a one-hot sector, so we might as well want to try to use the following relaxed likelihoods 
$$
p(y \vert z, x) = \mathcal{N}(y \vert \sum_k z_k\mu_k(x), \sum_k z_k\sigma_k(x)^2)
$$
or
$$
p(y \vert z, x) = \sum_k z_k \mathcal{N}(y \vert \mu_k(x), \sigma_k(x)^2).
$$
In the context of rider measurements we should have \mu and \sigma independent of x, that is 
$$
p(y \vert z, x) = \mathcal{N}(y \vert \sum_k z_k\mu_k, \sum_k z_k\sigma_k^2)
$$
and
$$
p(y \vert z, x) = \sum_k z_k \mathcal{N}(y \vert \mu_k, \sigma_k^2).
$$


### Joint model and posterior approximation

The joint model is then
$$
p_{\theta}(y, z \vert x) = p_{\theta}(y \vert z, x) p_{\theta}(z \vert x)
$$
Where \theta collects the parameters of the likelihood (decoder) and the prior.  We will approximate the posterior p(z \vert x, y)  with a q_{\phi}(z \vert y, x)   which is also a concrete distribution  with parameters \pi^{q}_{k}(y,x). 
$$
\pi^{q}_{k}(y,x)
$$

$$
q_{\phi}(z \vert y, x) 
$$


### Loss / negative ELBO

The variational free energy  / negative ELBO (loss) to optimise, is then is
$$
L(\theta, \phi) = \sum_i -E_{q}[\log p(y_i \vert z_i, x_i)] -E_{q}[\log p(z_i \vert x_i)] + E_{q}[\log q(z_i \vert y_i, x_i)]
$$
Which we approximate by MC sampling
$$
\begin{align}

\end{align}
$$
