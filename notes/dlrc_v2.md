# Models

- Gaussian conditioned on state

$$
\begin{align}
p(Y \vert X) &= \prod_i p(y_i \vert x_i)
\\
& = \prod_i\prod_jN(y_{ij}^{lidar} \vert \mu_{j}^{lidar}(x_i), \sigma_j^{lidar}(x_i)^2) \times \prod_{s,t}N(y^{depth}_{i,st} \vert \mu^{depth}_{st}(x_i),\sigma^{depth}(x_i)^2 )
\end{align}
$$

- Mimicking conditional Gaussians
  $$
  \begin{align}
  p(Y \vert X) &= \prod_i p(y_i \vert x_i)
  \\
  &= \prod_i p(y_i^{lidar} \vert x_i)\times p(y_i^{depth} \vert x_i)
  \\ 
  & = \prod_i\prod_jN(y_{ij}^{lidar} \vert \sum_k \pi_{jk}^{lidar}(x_i)\mu_{jk}^{lidar}(x_i), \sum_k \pi_{jk}^{lidar}(x_i)\sigma_{jk}^{lidar}(x_i)^2) 
  \\&\times \prod_{s,t}N(y^{depth}_{i,st} \vert \sum_k \pi_{st,k}^{depth}(x_i) \mu^{depth}_{st,k}(x_i), \sum_k \pi_{st,k}^{lidar}(x_i)\sigma^{depth}_{k}(x_i)^2 )
  \end{align}
  $$




  loss
$$
L(Y, X, \lambda)= \sum_i -\log p(y_i \vert x_i) - \lambda \sum_i\sum_k\pi_k(x_i) \log \pi_k(x_i) -\gamma \sum_i\sum_k c_{ik} \log \pi_k(x_i)
$$
  Loss autoencoder


$$
L(Y, X, \lambda)= \sum_i -\log p(y_i \vert x_i) - \lambda \sum_i\sum_k\pi_k(x_i) \log \pi_k(x_i) -\gamma \sum_i\sum_k c_{ik} \log \pi_k(x_i)
$$

.                                                                                                                        -