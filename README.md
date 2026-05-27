# coregrad-mnist

This repository contains an end-to-end training and evaluation pipeline for a handwritten digit classifier, implemented completely from scratch using the `coregrad` scalar-based automatic differentiation engine.

The primary objective of this project is to empirically validate the computational stability, correctness, and optimization capability of `coregrad` when training parameterized models within high-dimensional non-linear decision spaces.

---

## CoreGrad Dependency

This repository is built entirely on top of the `coregrad` scalar-based automatic differentiation engine.

CoreGrad implements a minimal reverse-mode automatic differentiation system with dynamic computational graph construction, scalar-level gradient propagation, and topologically ordered backpropagation.

Repository: https://github.com/Himanshu7921/coregrad

The primary purpose of this project is to empirically validate the correctness, numerical consistency, and optimization capability of the `coregrad` autodiff engine under real neural network training workloads involving high-dimensional non-linear classification tasks.

All gradient propagation, parameter updates, computational graph traversal, and optimization dynamics in this repository are executed exclusively through `coregrad`.

## Repository Objective

This implementation bypasses high-level tensor abstractions to study neural network optimization dynamics at the most granular level. The codebase is designed to evaluate:

* **Adjoint Propagation Stability:** Verifying that reverse-mode automatic differentiation scales reliably across thousands of sequential scalar graph operations.
* **Optimization Trajectory:** Tracking gradient descent convergence patterns using pure scalar-level partial derivatives.
* **Architectural Modularity:** Constructing dense linear layers, activation blocks, and cross-entropy loss tracking primitives directly out of individual `Scalar` nodes.

---

## Pipeline Components

The repository includes:

* **Data Ingestion:** Modular pipeline to stream, flatten, and normalize raw MNIST handwritten digit vectors.
* **Network Graph Definition:** Multi-layer feedforward perceptual network constructed entirely using `coregrad.Scalar` nodes.
* **Optimization Loop:** Batch Stochastic Gradient Descent (SGD) execution trace utilizing localized automated backpropagation (`.backward()`).
* **Evaluation Metrics:** Verification scripts calculating validation accuracy, cross-entropy loss decay, and gradient distribution matrices.

---

## Dependencies

* `coregrad==0.0.3`
* `numpy` *(For vectorized data loading and matrix-to-scalar parsing utilities)*

---

## Author

**Himanshu Singh** Research Engineer

*Year: 2026*