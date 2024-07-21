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

### Project 2: Conditional Gradient for Dual SVM and Subgradient for Phase Retrieval

**Objective:**
This project involves implementing the conditional gradient algorithm for the dual Support Vector Machine (SVM) problem and the subgradient algorithm for phase retrieval. The focus is on deriving mathematical solutions and implementing algorithms to optimize functions in data science.

**Key Components:**
1. **Conditional Gradient for Dual SVM:**
   - **Dual SVM Problem:** Formulation of the dual problem for SVM to find a hyperplane that maximizes the minimum distance to both classes.
   - **Gradient Derivation:** Detailed derivation of the gradient for the dual SVM objective with respect to the variable \(\alpha\).
   - **Linear Minimization:** Derivation of the closed-form solution for minimizing a linear function over the constraints.
   - **Step Size Rule:** Implementation of Cauchy's rule for determining the step size in the conditional gradient algorithm.
   - **Algorithm Implementation:** Testing the conditional gradient algorithm on a simple OR dataset and plotting the resulting hyperplane.

2. **Subgradient for Phase Retrieval:**
   - **Phase Retrieval Problem:** Formulation of the phase retrieval problem as minimizing the difference between the squared magnitude of linear measurements and target values.
   - **Convexity Analysis:** Examination of the convexity of the objective function.
   - **Subgradient Calculation:** Derivation of the subdifferential for the absolute function and the gradient for the phase retrieval objective.
   - **Data Generation:** Script to generate synthetic datasets for testing the phase retrieval algorithm.
   - **Algorithm Implementation:** Testing the subgradient algorithm on generated datasets and analyzing convergence using log error plots.

**Implementation:**
- **Conditional Gradient Algorithm:** Implemented and tested on the OR dataset, with results showing the hyperplane found via the dual SVM problem.
- **Subgradient Algorithm:** Implemented and tested on synthetic datasets with varying sizes and noise levels, analyzing the convergence of the algorithm under different conditions.

**Results:**
- **Hyperplane Visualization:** Conditional gradient algorithm effectively finds the separating hyperplane for the OR dataset.
- **Convergence Analysis:** Subgradient algorithm shows stable convergence for larger datasets and improved stability using an upper-bounded \( f^* \).

**Conclusion:**
The project provides an in-depth analysis of conditional gradient and subgradient methods for optimization problems in data science. The implementation demonstrates practical applications of these methods, highlighting their effectiveness and limitations.

### Project 3: Dual Averaging for SVM and Accelerated Gradient Descent

**Objective:**
This project focuses on implementing dual averaging and accelerated gradient descent algorithms to solve variations of Support Vector Machines (SVM) problems. It aims to derive theoretical foundations, implement algorithms, and perform experiments to analyze their performance.

**Key Components:**

1. **Dual Averaging for SVM:**
   - **Algorithm Proposal:** Introduction of Mirror Descent as a solution for the given optimization problem, highlighting its convergence guarantees.
   - **Implementation:** Developing the dual averaging algorithm for a specific SVM variation with constraints on the norm of the weight vector.
   - **Experiments:** Running the implementation on a provided dataset and reporting the final hyperplanes for different constraint values.

2. **Accelerated Gradient Descent for Squared Hinge SVM:**
   - **Gradient Derivation:** Detailed derivation of the gradient for the squared hinge loss SVM objective.
   - **L-smoothness Parameter:** Estimation of the L-smoothness parameter for the objective function.
   - **Algorithm Implementation:** Implementing the accelerated gradient descent algorithm with various step sizes and types, including regular gradient descent, accelerated gradient descent, and an optimized version.
   - **Experiments:** Running the algorithm on a default dataset and reporting results for different regularization parameters.

**Implementation:**
- **Dual Averaging Algorithm:** Implemented and tested on a dataset with variations in the constraint parameter \( c \).
- **Accelerated Gradient Descent Algorithm:** Implemented with different configurations and tested on the default dataset. The experiments include normalizing features and transforming labels for consistency.

**Results:**
- **Dual Averaging:** Visualization of hyperplanes for different constraint values, showing the effect of the constraint on the SVM solution.
- **Accelerated Gradient Descent:** Plots of function values and gradient norms on training and test sets for different algorithms and regularization parameters, demonstrating the convergence behavior and performance differences.

**Conclusion:**
The project provides a comprehensive analysis of dual averaging and accelerated gradient descent methods for SVM optimization problems. The implementation and experiments highlight the practical applications of these methods and their effectiveness in different scenarios.

### Project 4: Optimization Algorithms for Solving Sudoku Puzzles

**Objective:**
This project applies various optimization algorithms to solve Sudoku puzzles. It explores alternating projection, Douglas-Rachford splitting, and relaxation techniques for handling Sudoku constraints in different encoding formats.

**Key Components:**

1. **First Attempt: Alternating Projection for Integer Encoding**
   - **Projection to Constraints:** Develop a method to project a vector to satisfy row, column, and block constraints, ensuring numbers appear only once in each segment.
   - **Algorithm Implementation:** Implement an alternating projection algorithm that iteratively adjusts rows, columns, and blocks to meet Sudoku constraints.
   - **Solution Verification:** Test the algorithm on provided Sudoku puzzles and verify the solutions.

2. **Second Attempt: Douglas-Rachford Splitting for Binary Encoding**
   - **Binary Tensor Representation:** Encode the Sudoku puzzle as a binary tensor, where each entry indicates the presence of a number.
   - **Projection to Permutation Matrix:** Implement the Hungarian algorithm to find the closest permutation matrix to a given matrix.
   - **Douglas-Rachford Algorithm:** Implement the Douglas-Rachford splitting algorithm to iteratively solve the Sudoku puzzle, ensuring constraints are met for rows, columns, and blocks.

3. **Third Attempt: Relaxation to Doubly Stochastic Matrices**
   - **Row-Stochastic Matrix Projection:** Develop a method to find the closest row-stochastic matrix to a given matrix using KL divergence minimization.
   - **Doubly Stochastic Matrix Projection:** Extend the method to find the closest doubly stochastic matrix, projecting alternately onto row and column constraints.
   - **Algorithm Implementation:** Implement an alternating projection algorithm for Sudoku puzzles with fractional encoding, followed by a Dykstra’s algorithm for finding the closest doubly stochastic matrix.

**Implementation:**
- **Alternating Projection Algorithm:** Implemented for integer encoding, ensuring the preservation of given values and projecting onto row, column, and block constraints.
- **Douglas-Rachford Splitting Algorithm:** Implemented for binary encoding, handling the projection onto row, column, and block constraints in parallel.
- **Relaxation Algorithm:** Implemented for finding row-stochastic and doubly stochastic matrices, with applications to Sudoku constraints using alternating projection and Dykstra’s algorithm.

**Results:**
- **Integer Encoding:** The alternating projection algorithm effectively solves the Sudoku puzzles, meeting all constraints.
- **Binary Encoding:** The Douglas-Rachford splitting algorithm converges to valid Sudoku solutions, with efficient handling of constraints.
- **Relaxation Approach:** The relaxation to doubly stochastic matrices provides an alternative solution method, demonstrating the versatility of optimization techniques in solving Sudoku puzzles.

**Conclusion:**
The project demonstrates the application of advanced optimization algorithms to solve Sudoku puzzles. Each method offers unique advantages, with practical implementations showcasing their effectiveness in meeting Sudoku constraints.

### Project 5: Advanced Optimization Algorithms in Machine Learning

**Objective:**
This project explores advanced optimization algorithms for machine learning, specifically focusing on stochastic gradient-descent-ascent with variance reduction and randomized smoothing techniques. The goal is to derive theoretical solutions, implement algorithms, and analyze their performance on given datasets.

**Key Components:**

1. **Stochastic Gradient-Descent-Ascent with Variance Reduction:**
   - **Lagrangian Formulation:** Analyzing the Lagrangian formulation of a machine learning algorithm and deriving the optimal maximizer \(\alpha\) and the primal objective function.
   - **Dual Objective Function:** Deriving the optimal minimizer \(w\) and the dual objective function \(f(\alpha)\).
   - **Algorithm Implementation:** Implementing the stochastic gradient-descent-ascent algorithm and deriving gradients and projections.
   - **Experiments:** Running the algorithm on a default dataset with different step sizes (\(\eta_t = 1/t\) and \(\eta_t = 1/\sqrt{t}\)) and analyzing the convergence through multiple experiments.

2. **Randomized Smoothing:**
   - **Coordinate Gradient Algorithm:** Developing and analyzing a randomized coordinate gradient algorithm to solve the SVM variant.
   - **Gradient Approximation:** Implementing stochastic gradient algorithms with gradient approximations using smoothing parameters.
   - **Experiments:** Running the algorithm on the default dataset with specified smoothing parameters and step sizes, and plotting the results.

**Implementation:**
- **Stochastic Gradient-Descent-Ascent Algorithm:** Implemented with variance reduction techniques, tested on the default dataset, and analyzed for different step sizes. The implementation includes:
  - Derivation of optimal maximizer \(\alpha\) and minimizer \(w\).
  - Pseudo-code and actual implementation for stochastic gradient-descent-ascent.
  - Analysis of convergence through multiple iterations and runs.
- **Randomized Smoothing Algorithm:** Implemented and tested on the default dataset, focusing on:
  - Randomized coordinate gradient algorithm.
  - Gradient approximations using different smoothing techniques.
  - Visualization of convergence results through mean and standard deviation curves.

**Results:**
- **Stochastic Gradient-Descent-Ascent:** Analysis of function values and convergence rates for different step sizes, demonstrating the impact of step size on algorithm performance.
- **Randomized Smoothing:** Visualization of the convergence behavior for different gradient approximations, showing the effectiveness of smoothing techniques in optimization.

**Conclusion:**
The project provides an in-depth exploration of advanced optimization algorithms in machine learning. The implementation and experiments demonstrate the practical applications of these methods and their impact on optimizing machine learning models.

