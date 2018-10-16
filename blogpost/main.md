# Unsupervised Perception of Dynamic Environments

* Team: Boring Panda
* Members: Daniel, Giorgio

When we started to work on this project, one month ago, we decided to focus on the Machine Learning part. And reading some papers, a simple observation arose naturally:

Machine Learning for Robotics doesn't (yet) work and, when works, it works for specif cases well tuned. Why? How can we fix this problem?

More in general and related to the Robotic field, we asked us this question:

**Is it possible to sense a dynamic environment using learning techniques?**

Before exploring our solution to this problem, we need to understand the difference between a Geometry based approach and a Learning based approach.

## Geometry based and Learning based approach

The Robotic field has been historically reluctant to use Machine Learning techniques. With the advent of Deep Learning, the situation isn't improved. And it's easy to understand why! Typically in Robotics we deal with tasks well defined. Applying Geometry, Control theory and a lot of engineering, it's possible to obtain impressive practical results: why should we use methods that are approximated, without convergence theorems, and somehow obscures?

| ![](./images/geometry_based.jpg)                             | ![](./images/learning_based.jpg)                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Geometry based approach: we build a specific model of our system, and on top of it we use machine learning to solve specific tasks. | Learning based approach: we build a statistical model, and on top of it we inject geometrical constraints and task dependent informations. |

Difficult to debate with them, I agree. No need for Machine Learning, we only need curve fitting for specific tasks; we have a sufficient amount of Machines already. So done? Not so fast! 

Let's say that the task is not so well defined; let's say that we don't have perfect knowledge of our robotic agents; and in particular, let's say that we live in a Dynamic Environment. For this scenario, classical methods are not sufficient. Now Machine Learning can help.

### The path toward statistical modeling

So we decided to explore the possibility to perceive a Dynamic Environment using Machine Learning.

It was immediately clear that a classical Supervised approach was not well suited for this task: discriminative algorithms assume same data distribution between train and test set (it is possible to work with Domain Adaptation techniques, but they assume always some similarity in terms of distribution). And they output a number, something not adapt to a dynamic scenario.

So we started to think to use **Statistical Unsupervised Learning**, a form of ML that learns structure in data without manual labeling. The idea was nice. With one model we could be able to sense the environment without using heuristics to label sensor data and quantify uncertainty around our prediction. However, we are trying to predict something in an Environment that changes, and we don't know how it will change!

## Preliminary results

After some brainstorming with our mentor, the path was clear: we decided to build a sensing framework, using unsupervised learning techniques, in a statistical setup.

Instead to conjecture how to do, we decided to start collecting data and seeing if it was possible to detect some patterns. The first result was cheering: collecting lidar data on a given trajectory, we started to see a clear multi modal pattern.

TODO: RESIZE TO HALF THE ORIGINAL DIMENSION

|           ![](./images/jointplot_trajectory0.png)            |
| :----------------------------------------------------------: |
| Data collected moving the robotic arm on a given trajectory. For any lidar we show the temporal trace in mm and the relative histogram with density estimation. |

| ![](./images/measurements_lidar3_with_thresh.png)            |
| ------------------------------------------------------------ |
| Controlled experiment to analyze different patterns for background, self and other. |

So Multi modality? Why not try with a Mixture Model? Mixture Models are a classical tool for statistical clustering and they seemed a good and simple idea for our problem. In general a Mixture Model can be defined as:
$$
\begin{align}
p(y) &= \sum_k \pi_k \mathcal P_k(y \vert \mu, \sigma)
\end{align}
$$
So we tried a simple Gaussian Mixture Model on a simplified scenario: we collected time-series data in static and dynamic environment; every sample is a 10 consecutive points in this time series; and we built a low dimensional handcrafted feature representation. In particular, given a time series x(t), we built a simple handcrafted feature representation using [max;std] of the following simple (non-linear) data transformation. We can think to this f(x_t) as a temporal difference
$$
f(x_t) = \vert x(t+1) - x(t) \vert
$$
the result was good: considering 30 samples (every sample consisting of short time series with 10 consecutive points), we were able to cluster correctly 70 % of these points. And the misclassified one are the one dynamic one indistinguishable from the static one in this embedding space. 

| ![](./images/gt_gmm.png)                                     | ![](./images/pred_gmm.png) |
| ------------------------------------------------------------ | -------------------------- |
| time series data embedding for dynamic (red) and static (blue) environment | gmm clustering prediction  |

Good! The preliminary results showed that, in principle, it's possible to cluster these time series. And given that we are able to cluster with handcrafted features, using deep learning and statistics, we should be able to find a representation space to solve our perception task. 

It's time to build our model, that we, um-modestly, decided to call **Sensing Framework**. 



## Sensing Framework

The sensing framework is a hierarchical model and consists of three main Modules:

* Anomaly Detection Module
* Clustering Module
* Collision Detection Module

These three modules solve what we consider the most relevant perception tasks for a manipulator in Dynamics Environment: understand if there is something new or dynamic in the environment; if everything is normal, be able to distinguish between the background and itself; instead, if something is abnormal, distinguish between simply something new (agent moving, new object) and a possible collision.

As input for all the framework we have the 9 lidar readings, and the 7 joint position (we decide to not use joint velocity and torque because not necessary and not always available).



| ![](./images/framework.jpg)                                  |
| ------------------------------------------------------------ |
| Pictorial view of our Perception Framework. Given input data sensor, we build a hierarchical learning pipeline to deal with complex Dynamic Environments. |

### Anomaly Detection Module

For anomaly detection we decided to use a Regression model. The model is a Normal model. The idea is to solve a proxy task (prediction) to obtain an anomaly detector: after training, when the model is not able to reconstruct or predict a given sample point, we consider this point an anomaly. Simple and effective. We used a multilayer perceptron with two 256 unit layers, hyperbolic tangent as activation. This network outputs two objects: 9 means (one for every lidar), and more important, 9 standard deviations. The network is trained minimizing the negative log likelihood. In this way, we can not only detect anomaly, but we quantify the uncertainty (building a confidence interval) for our prediction and we use this confidence interval to decide if a point is an anomaly or not. We don't need to use heuristics, because the probability to  be an outlier is directly linked with how well the model fits the data. 
$$
\begin{align}p(Y \vert X) 
&= \prod_i p(y_i \vert x_i)\\
&= \prod_i\prod_j \mathcal N(y_{ij}^{lidar} \vert \mu_{j}^{lidar}(x_i), \sigma_j^{lidar}(x_i)^2) \times \prod_{s,t} \mathcal N(y^{depth}_{i,st} \vert \mu^{depth}_{st}(x_i),\sigma^{depth}(x_i)^2 )
\end{align}
$$
$$
L(Y, X)= - \sum_i \log p(y_i \vert x_i)
$$


|                ![](./images/anomaly_blog.png)                |
| :----------------------------------------------------------: |
| anomaly detection in action. Lidar n.3 reading. We inject anomalies in the environment, and the model is not able to predict the new behavior. |

###  Clustering Module

If the Anomaly Detector Module detects an anomaly, we have done. But, if it detects a normal behavior, now we need to use the Clustering Module to decide if it is background or self. To obtain this result we built what we called a **selector model**, a network that mimics the Gaussian mixture model' s behavior. Again we solve a proxy task (a prediction) to solve a clustering task. We train with maximum likelihood and an entropy regularization term.

we assume the 9 lidars independent and identically distributed and we build a probabilistic model to fit the data. In particular we have interest in learning the moments of 9 one dimensional multi modal Gaussian with two modes: self and background. 

We input short time series of 10 consecutive points: in this way, we help the model to learn the dynamics and we filter noise, smoothing the prediction.

$$
\begin{align}
p(Y \vert X) 
&= \prod_i p(y_i \vert x_i)\\
&= \prod_i p(y_i^{lidar} \vert x_i)\times p(y_i^{depth} \vert x_i)\\ 
&= \prod_i\prod_j \mathcal N(y_{ij}^{lidar} \vert \sum_k \pi_{jk}^{lidar}(x_i)\mu_{jk}^{lidar}(x_i), \sum_k \pi_{jk}^{lidar}(x_i)\sigma_{jk}^{lidar}(x_i)^2) \\
&\times \prod_{s,t} \mathcal N(y^{depth}_{i,st} \vert \sum_k \pi_{st,k}^{depth}(x_i) \mu^{depth}_{st,k}(x_i), \sum_k \pi_{st,k}^{lidar}(x_i)\sigma^{depth}_{k}(x_i)^2 )
\end{align}
$$

$$
L(Y, X, \lambda, \gamma)= -\sum_i \log p(y_i \vert x_i) - \lambda \sum_i\sum_k\pi_k(x_i) \log \pi_k(x_i) -\gamma \sum_i\sum_k c_{ik} \log \pi_k(x_i)
$$


|                  ![](./images/cl_blog.png)                   |
| :----------------------------------------------------------: |
| clustering on a controlled experiment: lidar n.3 reading. The red vertical lines represent the limit of ground truth for self |

With this model, on a ground truth of 3000 points manually labeled, we obtained a global accuracy of 89.8 % and a recall (on the class of interest self) of 69.4 %.

$$

$$

### Collision Detection Module

We model this task as a simple multi-label binary classification. Given our Unsupervised approach (i.e. we don't label data), how can we train a classifier? Well thanks to a smart trick that we called negative sampling: In practice, we consider batches of 10 consecutive time series point; we randomly choose columns and set that values to Gaussian noise around a small value (let's say 50 mm). We label any original data point class 0, and any modified data point 1: in this way we want to discriminate between a generic anomaly (something new in the environment, other agent acting) and a probable collision. We now train/test on this dataset and we evaluate the result considering a global result and a per lidar result (we solve a binary classification problem for every lidar). Computing the Confusion matrix for a test set sample of 2000 points,  we evaluate the classification result using the Jaccard index and the F1 metric. 

We report results for the class of interest (collision or class 1):



| metrics/Lidar | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| N             | 100  | 180  | 140  | 180  | 100  | 170  | 200  | 160  | 180  |
| Sensitivity   | 96.0 | 33.3 | 22.1 | 56.1 | 84.0 | 97.0 | 22.0 | 20.6 | 57.2 |
| IoU           | 92.3 | 32.6 | 20.0 | 50.5 | 73.6 | 91.6 | 19.5 | 17.5 | 50.2 |
| F1            | 96.0 | 49.1 | 33.3 | 67.1 | 84.8 | 95.6 | 32.7 | 29.8 | 66.8 |



## Conclusions

In this work, we built an unsupervised framework for sensing the environment without relying on a fixed configuration or geometry. This approach has Advantages and Drawbacks.

Regarding the advantages, the most obvious, and the goal of this project, is to learn a model that can be transferred to different scenarios with minimum fine-tuning; another pro, less obvious, is the capacity to quantify the uncertainty around prediction in an intuitive and immediate way, because the approach is intrinsically statistical.

But there are drawbacks: the most relevant of them is that we are not able to interact easily with the environment: we can sense, we can learn a model of the environment; but also with this knowledge is not straightforward how to use this knowledge and solve a specific task.

The model can obviously be improved: in particular we can use latent variable models to learn better representation and sequence models to model the temporal dynamics.

