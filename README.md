# MSc-Thesis-Project-Quantum-Properties-on-Crystals-by-Ultrafast-Non-linear-Pumping


This repository contains the analytical and numerical work related to the study of nonlinear optical response and harmonic decomposition of reflected intensity on crystals such as Tellurium. The goal is to study the coherent phonon oscillations in the crystals, excited by ultrafast pumping, and connect the analytical expressions derived from nonlinear susceptibilities with numerical analysis of the experimental data collected by pump-probe spectroscopy.

The project combines symbolic derivations (Mathematica) with numerical data processing and visualization using Python.

---

# Repository Structure

```
project-root/
│
├── README.md
│
├── analytical/
│   ├── mathematica/
│   │   ├── SHG-reflected-intensity-factorization.nb
│   │   
│   │
│   └── derivations/
│       ├── reflected_intensity_derivation.pdf
│     
│
├── numerics/
│   ├── scripts/
│   │   ├── Te_ultrafast_analysis.py
│   │
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   │
│   └── figures/
│       ├── polar_plots/
│       └── harmonic_fits/
│
└── docs/
    ├── references/
    └── notes.md
```

---

# Analytical Work

The **analytical** part of the project focuses on symbolic manipulation of nonlinear optical expressions.

This includes:

* Expansion of the reflected intensity at linear a second-order.
* Identification of harmonic components.
* Factorization of coefficients depending on the first and second-order susceptibilities.
* Connection between harmonic amplitudes, nonlinear susceptibilities, and coherent phonon oscillations.

Tools used:

* **Mathematica**
* symbolic tensor manipulation
* complex conjugation and factorization
* harmonic decomposition

The Mathematica notebooks are located in

```
analytical/mathematica/
```

Typical tasks in this folder include:

* expanding intensity expressions
* grouping terms into (I_0), (I_4), (I_8)
* factorizing susceptibility combinations
* verifying symmetry relations

---

# Numerical Work

The **numerical** section contains scripts used to analyze experimental data.

Main tasks include:

* extracting harmonic components from measured intensity
* fitting angular dependence
* phase alignment of polarization axes
* visualization in polar coordinates

Scripts are stored in

```
numerical/scripts/
```

Data organization:

```
numerical/data/raw
```

raw experimental data

```
numerical/data/processed
```


Generated plots and figures are stored in

```
numerical/figures/
```

---


# Requirements

Numerical analysis may require:

* Python 3
* numpy
* scipy
* matplotlib

Symbolic work requires:

* Wolfram Mathematica

---

# Future Extensions

Possible future additions include

* automated harmonic extraction pipelines
* comparison with theoretical models
* parameter fitting for nonlinear susceptibilities
* integration with experimental datasets
* **lab flow-structure (raw → clean → analysis → figures)**
