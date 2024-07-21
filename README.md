# CO673/CS794 Fall 2022 - Optimization in Data Science

[Course website](https://cs.uwaterloo.ca/~y328yu/mycourses/794/index.html)

**University of Waterloo**

---

**Course Overview:**
The course "Optimization in Data Science" aims to provide a comprehensive introduction to essential optimization techniques and theories specifically tailored for data science applications. It focuses on vector spaces, matrices, norms, gradients, Hessians, convex sets, functions, and various gradient algorithms. The course emphasizes both theoretical foundations and practical applications in machine learning and artificial intelligence.

**Course Objectives:**
- Understand vector spaces, including addition and scalar multiplication.
- Explore matrices, matrix transposition, multiplication, and the concept of eigenvalues and eigenvectors.
- Learn about different norms such as L1, L2, and Infinity norms.
- Study convex sets and functions, and Fermat's optimality condition for unconstrained optimization.
- Master gradient descent algorithms and their variants, including Stochastic Gradient Descent (SGD), Mini-batch Gradient Descent, Momentum, Nesterov Accelerated Gradient (NAG), AdaGrad, RMSProp, and Adam.

### Project 1: Orthogonal Residual Polynomial Iteration and Gradient Descent for Huber Regression

**Objective:**
This project explores polynomial iterations, particularly the conjugate gradient algorithm, and applies gradient descent techniques to Huber regression. It involves theoretical derivations and practical implementations to optimize functions in data science.

**Key Components:**
1. **Orthogonal Residual Polynomial Iteration:**
   - **Derivation of Three-term Recursion:** Detailed derivation of polynomial iterations and their remarkable properties, including connections to reproducing polynomial kernels.
   - **Orthogonal Polynomials:** Analysis of orthogonal residual polynomials through three-term recursion and verification using inner products.

2. **Gradient Descent for Huber Regression:**
   - **Huber Regression Objective:** Minimize the Huber regression objective using gradient descent with two step-size choices: constant and Armijo's backtracking rule.
   - **Gradient Calculation:** Derivation of the gradient for the Huber regression objective.
   - **L-smoothness:** Evaluation of whether the objective function is L-smooth with respect to the Euclidean norm.

**Implementation:**
- **Gradient Descent Algorithm:** Implementation of gradient descent with constant and backtracking step sizes.
- **Application on Boston Housing Dataset:** Applied gradient descent to train on the Boston housing dataset, analyzing the performance on the test set using average error metrics.

**Results:**
- **Convergence Analysis:** Comparison of convergence rates for constant and backtracking step sizes.
- **Performance Metrics:** Visualization of objective function values and average errors over iterations for both step-size methods.

**Conclusion:**
The project provides a comprehensive analysis of polynomial iterations and gradient descent methods for optimization in data science. The implementation showcases the practical application of these methods, highlighting the importance of choosing appropriate step sizes for convergence.


