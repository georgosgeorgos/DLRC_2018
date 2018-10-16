# Models

- Loss  semisupervision + autoencoder_depth 



$$
L(Y, X, \lambda)= \sum_i -\log p(y_i \vert x_i) - \lambda \sum_i\sum_k\pi_k(x_i) \log \pi_k(x_i) -\gamma \sum_i\sum_k c_{ik} \log \pi_k(x_i) + L(x_{depth} - \hat x_{depth})
$$

model with selector



$$
\begin{align}
p(Y \vert X) &= \prod_i p(y_i \vert x) \\
             &= \prod_i N(y_{i} \vert \sum_k \pi_{ik}(x_i)\mu_{ik}(x), \sum_k \pi_{ik}(x)\sigma_{ik}(x)^2) 
\end{align}
$$

explicit GMM?

$$
\begin{align}p(Y \vert X) &= \prod_i p(y_i \vert x) \\             &= 
\prod_i [\sum_k \pi_{ik}(x) N(y_{i} \vert \mu_{ik}(x), \sigma_{ik}(x)^2)] \end{align}
$$




















