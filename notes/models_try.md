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


TODO:
-	collect very large dataset at 100Hz, at least 4 hours of data
-	try all models with and without entropy regularisation that is in toal 9 models: Model 0 and 2 versions of models 1.1,1.2, 2.1, 2.2


