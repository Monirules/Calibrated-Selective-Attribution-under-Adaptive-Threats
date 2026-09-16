<div align="center">

# 🕵️ Know When to Blame
### Calibrated Selective Attribution for LLM Forensics and the Limits of Adaptive Evasion

<br>

[![Submitted to NAACL 2027](https://img.shields.io/badge/Submitted%20to-NAACL%202027-8A2BE2?style=for-the-badge&logo=readthedocs&logoColor=white)](https://2027.naacl.org)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#)
[![Conformal Prediction](https://img.shields.io/badge/Conformal-Prediction-2B6CB0?style=flat-square)](#)
[![LLM Security](https://img.shields.io/badge/LLM-Security-C44E52?style=flat-square)](#)
[![Made with Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white)](#)

<br>

*A calibration layer that lets an LLM Forensic tool say* ***"I am not sure"*** *instead of blaming the wrong source file.*

</div>

---

## 🎯 The Problem in One Picture

When someone attacks a large language model, for example by hiding a malicious instruction inside a document (prompt injection) or by poisoning the knowledge base of a RAG system, a **forensic tool** tries to point at *which piece of text caused the attack*. Two strong recent tools do this: **AttnTrace** (attention based) and **RAGOrigin** (loss + retrieval based).

The catch: these tools **always answer**, even when they are unsure. A confident but wrong answer means blaming an innocent document, which is worse than saying nothing. Our work adds a simple, provable "know when to blame" gate on top of any such tool.

---

## 💡 What We Built

> **CSA (Calibrated Selective Attribution)** — a thin statistical wrapper that keeps only the *confident* answers and abstains on the rest, with a **mathematical guarantee** on how often it is wrong when it does answer.

<table>
<tr>
<td width="50%" valign="top">

### 🧩 CSA (the defense)
- Uses **split conformal prediction** to pick a confidence threshold on a calibration set.
- Answers only when the attribution "gap" score clears the threshold; otherwise it **abstains**.
- Comes with a finite sample bound: **P(wrong | answered) ≤ α / coverage**, with α = 0.10.
- Works as a plug in layer on top of AttnTrace *or* RAGOrigin, no retraining.

</td>
<td width="50%" valign="top">

### ⚔️ GapAttack (the stress test)
- An **adaptive attacker** whose only goal is to inflate the confidence score and sneak a wrong answer past the gate.
- **Mode A** — score space, a worst case upper bound.
- **Mode B** — *real text* edits (keyword distractors, duplication, zero width characters) under a fixed 40 edit budget.
- We report Mode B as reality and Mode A as the pessimistic ceiling.

</td>
</tr>
</table>

---

## 📊 Key Results

All numbers below are at **α = 0.10** (target error budget). Colors: 🟢 good / robust, 🔴 attack ceiling.

| Setting | Tool / Model | Accuracy (raw) | 🟢 CSA risk | Coverage | 🔴 GapAttack (Mode A, ceiling) | 🟢 GapAttack (Mode B, real text) |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| Prompt injection | **AttnTrace** / Llama | 0.777 | **0.099** | 0.81 | 0.243 | **0 / 150** |
| Prompt injection | **AttnTrace** / Qwen | 0.787 | **0.074** | 0.81 | 0.211 | **0 / 150** |
| RAG poisoning | **RAGOrigin** / Llama | 0.820 | **0.051** | 0.78 | 0.213 | **0 / 50** |

**Read this as:**

- ✅ **CSA works.** It pulls the wrong answer rate down to roughly **0.05 – 0.10** while still answering on about **78 – 81%** of cases. The bound holds.
- ✅ **Real text GapAttack fails.** Across **200 pooled real text attack attempts**, the attacker landed **0 successful wrong answers** (AttnTrace 95% upper bound ≈ 2.5%, RAGOrigin interval [0, 7.1%]).
- ⚠️ **The honest ceiling.** In the pure score space setting (Mode A), where the attacker is handed perfect control of the score, error can reach ~0.21 – 0.24. This is a *worst case bound*, not something we observed with real edits. We report both.

> 🔬 **Bonus finding:** for RAGOrigin, the native gap score is a weak signal (AUROC 0.55), but the 2-means centroid distance (`cdist`) recovers a strong one (**AUROC 0.89**). CSA is built on the stronger signal.

---

## 🧾 Honesty Note

This project follows a strict **no overclaiming** rule. Every number here is measured, not tuned for effect. Where a result is a pessimistic upper bound (Mode A) rather than an observed outcome, we say so. The RAGOrigin poisoning uses a medium stealth PoisonedRAG variant, which is documented as a limitation in the paper. If you find a number you cannot reproduce, please open an issue.

---

## 📁 Repository Structure

```
CSA_Project/
├── csa_utils.py                       # Conformal helpers (threshold fit, risk/coverage, AUROC, ECE)
├── CSA_Phase5b_ModeB.ipynb            # AttnTrace real-text GapAttack (Mode B)
├── CSA_Phase5c_RAGOrigin_ModeB.ipynb  # RAGOrigin real-text GapAttack (Mode B)
├── CSA_Finish.ipynb                   # Memory-safe RAGOrigin run + Qwen verification
├── CSA_Qwen_Recheck.ipynb            # Confirms 0 true Qwen successes (artifact fix)
├── figures/                           # Coverage–risk, AUROC, breach, false-alarm plots
├── records/                           # Per-case JSON results
├── CSA_Findings_and_Limitations.md    # Plain-English findings + limitations
└── paper/                             # NAACL 2027 submission (main2.tex, refs2.bib)
```

---

## 🚀 How to Run

```bash
# 1. Environment (Python 3.10+, one GPU is enough — tested on a 12 GB RTX 5070)
conda create -n csa python=3.10
conda activate csa
pip install torch transformers numpy scikit-learn matplotlib jupyter

# 2. Launch the notebooks
jupyter notebook

# 3. Reproduce the headline experiments
#    - CSA_Phase5b_ModeB.ipynb        -> AttnTrace  (prompt injection)
#    - CSA_Phase5c_RAGOrigin_ModeB.ipynb -> RAGOrigin (RAG poisoning)
#    - CSA_Finish.ipynb               -> Run-All finisher (~40 min)
```

> 💡 **Tip:** the finisher notebook is memory safe (document token cap + OOM guard). On a single 12 GB GPU the full RAGOrigin pass runs in about **7 minutes**.

---

## 🧠 Method at a Glance

<img width="1205" height="604" alt="image" src="https://github.com/user-attachments/assets/ac32717b-4716-4be6-83da-f9534c10eeaa" />


---

## 📖 Citation

If you use this work, please cite:

```bibtex
@inproceedings{mahmud2027csa,
  title     = {Know When to Blame: Calibrated Attribution for LLM Forensics and the Limits of Adaptive Evasion},
  author    = {Mahmud, M. I. and Zhan, J.},
  booktitle = {Proceedings of the 2027 Conference of the North American Chapter
               of the Association for Computational Linguistics (NAACL)},
  year      = {2027},
  note      = {Under review}
}
```

---

<div align="center">

### ✍️ Author

**M. I. Mahmud** · Ph.D. Researcher, University of Cincinnati
📧 monirul.mahmud9@gmail.com · 🌐 [monirules.github.io](https://monirules.github.io) · 🎓 [Google Scholar](https://scholar.google.com/citations?user=9pLzr9UAAAAJ)

<br>

*Built with a focus on trustworthy, honest, and reproducible AI security research.*

**⭐ If this project helps your work, please consider starring the repo. ⭐**

</div>
