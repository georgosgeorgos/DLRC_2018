# Models

- Model 0: Plain NN models

$$
\begin{align}
p(Y \vert X) &= \prod_i p(y_i \vert x_i)
\\
& = \prod_i\prod_jN(y_{ij}^{lidar} \vert \mu_{j}^{lidar}(x_i), \sigma_j^{lidar}(x_i)^2) \times \prod_{s,t}N(y^{depth}_{i,st} \vert \mu^{depth}_{st}(x_i),\sigma^{depth}(x_i)^2 )
\end{align}
$$

- Model 1.1: selector with state dep means and variances
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






- Model 1.2: selector with local means and variances
  $$
  \begin{align}
  p(Y \vert X) &= \prod_i p(y_i \vert x_i)
  \\
  &= \prod_i p(y_i^{lidar} \vert x_i)\times p(y_i^{depth} \vert x_i)
  \\ 
  & = \prod_i\prod_jN(y_{ij}^{lidar} \vert \sum_k \pi_{jk}^{lidar}(x_i)\mu_{jk}^{lidar}, \sum_k \pi_{jk}^{lidar}[\sigma_{jk}^{lidar}]^2) 
  \\&\times \prod_{s,t}N(y^{depth}_{i,st} \vert \sum_k \pi_{st,k}^{depth}(x_i) \mu^{depth}_{st,k}, \sum_k \pi_{st,k}^{lidar}(x_i)[\sigma^{depth}_{k}]^2 )
  \end{align}
  $$

- Model 2.1: GMM with state dep means and variances
  $$
  \begin{align}
  p(Y \vert X) &= \prod_i p(y_i \vert x_i)
  \\
  &= \prod_i p(y_i^{lidar} \vert x_i)\times p(y_i^{depth} \vert x_i)
  \\ 
  & = \prod_i\prod_j [\sum_k \pi_{jk}^{lidar}(x_i) N(y_{ij}^{lidar} \vert \mu_{jk}^{lidar}(x_i),\sigma_{jk}^{lidar}(x_i)^2) 
  \\&\times \prod_{s,t}[\sum_k \pi_{st,k}^{depth}(x_i) N(y^{depth}_{i,st} \vert \mu^{depth}_{st,k}(x_i), \sigma^{depth}_{k}(x_i)^2 )]
  \end{align}
  $$

- Model 2.2: GMM with local means and variances
  $$
  \begin{align}
  p(Y \vert X) &= \prod_i p(y_i \vert x_i)
  \\
  &= \prod_i p(y_i^{lidar} \vert x_i)\times p(y_i^{depth} \vert x_i)
  \\ 
  & = \prod_i\prod_j[ \sum_k \pi_{jk}^{lidar}(x_i) N(y_{ij}^{lidar} \vert \mu_{jk}^{lidar}, [\sigma_{jk}^{lidar}]^2) ]
  \\&\times \prod_{s,t}[\sum_k \pi_{st,k}^{depth}(x_i) N(y^{depth}_{i,st} \vert  \mu^{depth}_{st,k}, [\sigma^{depth}_{k}]^2 )
  \end{align}
  $$




Overall loss

$$
L(Y, X, \lambda)= \sum_i -\log p(y_i \vert x_i) - \lambda \sum_i\sum_k\pi_k(x_i) \log \pi_k(x_i) -\gamma \sum_i\sum_k c_{ik} \log \pi_k(x_i)
$$



## Negative sampling model for anomaly detection

Let $z_t$ be a collector of state related variables for example $z_t = (x_t, y_t^{lidars})$ or some RNN or LSTM cell summarising a short history $z_t = LSTM(x_{t-s:t}, y^{lidar}_{t-s:t})$ form $t-s$ to $t$. 

We prospose a multi-output classifier $p_{t,j}^{lidar} = f_j(z_t)$  using a Bernoulli loss for each lidar. These will classify normal vs. abnormal data at lidar $j$. For each lidar $j$ , we consider a class label $c_j^{lidar}$ with $c_j^{lidat} = 1$ for normal data and $c_j^{lidar} = 0$ for abnormal data. We construct the training data in the following way

- for all recorded data we  use $c_{t,j}^{lidar} =1 $
- we randomly select a few liar indices $J$ and replace  the recorded $y_{t,J}^{lidar}$ data with some type of noise.(Uniform, Gaussian) and set $c_{t, J}^{lidar}=0$ to mimic abnormal data at lidars $J$ 

This classification data is then used to train the classifier $f$.

  


