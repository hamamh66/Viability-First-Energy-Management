# Viability-First Energy Management for Islanded Microgrids

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/Viability-First-Energy-Management/blob/main/Viability_First_Energy.ipynb)

## Overview

This repository provides the reproducible computational companion to the manuscript:

**“Viability-First Energy Management for Islanded Microgrids: Closed-Form Resilience Certification and Pareto Dispatch under Renewable Drift.”**

The work develops a **viability-first energy-management framework** for islanded microgrids combining renewable generation, battery storage, and a ramp-limited dispatchable energy source. Instead of optimizing economic performance over all instantaneously feasible states, the proposed strategy first identifies the states from which critical-load service remains recoverable under the worst admissible renewable deficit. Economic dispatch is then restricted to this certified viable region.

The central analytical result is the nominal closed-form viability condition

\[
s \geq \frac{(d_{\mathrm{hi}}-g)^2}{2R},
\]

where:

- \(s\) is the battery state of charge,
- \(g\) is the dispatchable-source output,
- \(R\) is its ramp capability, and
- \(d_{\mathrm{hi}}\) is the worst-case net deficit.

This relation links **storage reserve, conversion capability, renewable uncertainty, and resilience** in a directly interpretable form.

## Main Contributions

The repository supports the manuscript's main methodological and numerical contributions:

- Closed-form viability kernel for a battery-supported islanded microgrid.
- Analytical recoverability boundary under a worst-case renewable/load deficit.
- Viability-preserving bang-bang and proportional feedback policies.
- Dynamic comparison with efficiency-greedy and fixed-reserve strategies.
- Exact zero-order-hold simulation with state-dependent ramp constraints.
- Sampled-data protection margin for digital EMS implementation.
- Finite-horizon economic MPC.
- Economic MPC with a **terminal viability constraint**.
- Monte-Carlo evaluation under stochastic renewable/load drift.
- Robustness analysis under deficit and ramp-rate model mismatch.
- Refined physical models including discharge efficiency, depth-of-discharge limits, start-up delay, bounded drought duration, and curtailable load.
- Reproducible generation of manuscript figures, tables, JSON configuration files, and summary results.

## Repository Contents

```text
.
├── Viability_First_Energy.ipynb   # Canonical reproducible notebook
└── README.md
```

The notebook is designed as the **single canonical computational pipeline** for the study.

## Notebook Structure

The notebook is organized into the following experiments:

| Section | Purpose |
|---|---|
| **E0** | Closed-form viability kernel and grid verification |
| **E1** | Design sensitivity to ramp capability and worst-case deficit |
| **E2** | Deterministic worst-case renewable drought |
| **E2′** | Protection–efficiency trade-off |
| **E3** | Monte-Carlo stochastic renewable/load drift |
| **E4** | Finite-horizon MPC and terminal viability constraint |
| **E5** | Robustness to plant/model uncertainty |
| **E6** | Refined physical plant with losses, DoD floor, and start-up delay |
| **A** | Analytical kernel variants |
| **B** | Sampled-data protection margin |
| **C** | Unit and regression tests |
| **D** | Reproducibility summary |

## Selected Reproducible Results

With the nominal configuration used in the notebook:

- Battery capacity: **20 kWh**
- Dispatchable-source capacity: **6 kW**
- Ramp capability: **1.5 kW/h**
- Worst-case deficit: **5 kW**
- EMS sampling interval: **0.02 h**
- Nominal viable fraction of the state space: **0.8843**
- Minimum viable reserve at \(g=0\): **8.333 kWh**
- Sampled-data protection requirement: **0.20 kWh**
- Monte-Carlo scenarios: **300**
- Random seed: **20250901**

In the 300 stochastic 24-hour scenarios:

- the **efficiency-greedy policy shed load in 100% of runs**;
- the **viable bang-bang policy shed no critical load**;
- the **viable proportional policy shed no critical load**;
- the fixed-reserve baseline also avoided shedding, but used more dispatchable energy on average.

The notebook also shows that a viability terminal constraint can eliminate short-horizon MPC shedding under the correctly modeled worst-case experiment.

## Running in Google Colab

Google Colab is recommended.

1. Open `Viability_First_Energy.ipynb`.
2. Click **Runtime → Run all**.
3. Authorize Google Drive when prompted.

By default, outputs are written to:

```text
/content/drive/MyDrive/Outputs/Viability_First_Energy/
```

with the structure:

```text
Viability_First_Energy/
├── Figures/
└── results/
```

The notebook produces manuscript figures as PDF files and numerical outputs as CSV, JSON, and text files.

To run in Colab without Google Drive, set:

```python
SAVE_TO_DRIVE = False
```

## Local Execution

The notebook can also be executed locally. Google Drive mounting is skipped automatically outside Colab.

The main dependencies are:

- Python
- NumPy
- SciPy
- pandas
- Matplotlib

A custom output directory can be supplied through the environment variable:

```bash
UVIF_OUT=/path/to/output
```

## Reproducibility

All principal experimental parameters are contained in immutable configuration objects inside the notebook. The computational workflow uses:

- deterministic seed `20250901`;
- one shared state integrator;
- explicit plant and experiment configurations;
- regression assertions on headline numerical results;
- exported parameters and tables;
- an automatically generated `reproducibility_summary.txt`.

The notebook is intended to allow the figures, tables, and principal numerical results reported in the manuscript to be regenerated from one source.

## Energy-Management Perspective

The proposed framework differs from conventional reserve policies because the required energy reserve is **state-dependent**. A fixed battery reserve ignores the current capability of the dispatchable conversion unit. Under viability-first management, the reserve required for resilience changes with the generator or fuel-cell output and its ramp capability.

This leads to an operational interpretation in which:

- battery energy represents stored recovery capacity;
- dispatchable-source ramp capability determines how quickly a deficit can be absorbed;
- renewable/load uncertainty determines the disturbance that must be defended;
- economic optimization is permitted only when recoverability remains certified.

The framework therefore connects **energy storage, energy conversion, renewable integration, real-time control, and resilient energy management**.

## Citation

If you use this repository, please cite the associated manuscript:

> Y. Y. Ghadi, R. Zayoud, I. Alreshidi, S. Guizani, and H. Hamam,  
> **“Viability-First Energy Management for Islanded Microgrids: Closed-Form Resilience Certification and Pareto Dispatch under Renewable Drift.”**

Publication metadata and DOI can be added here once available.

## Authors

- Yazeed Yasin Ghadi
- Rahma Zayoud
- Ibrahim Alreshidi
- Sghaier Guizani
- Habib Hamam

## License

A license should be selected before public release of the repository. For open research software, a standard permissive license such as **MIT** or **BSD-3-Clause** may be considered.

---

**Note:** Replace `USER` in the Colab badge URL with the GitHub account or organization name after the repository is created.
