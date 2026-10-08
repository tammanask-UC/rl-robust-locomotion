# Robust Locomotion Under Dynamics Shift: PPO vs. SAC with Domain Randomization

Reinforcement Learning Project, Part 2 · Suresh Tammana · October 2026

This project tests whether a locomotion policy trained in one simulator keeps working when the physics changes. It compares **PPO** and **SAC**, each trained under **nominal** physics and under **domain randomization**, on Gymnasium **Hopper-v5** (MuJoCo), and evaluates all 12 trained agents on a 5 × 5 grid of body-mass and friction changes (3,000 evaluation episodes).

## Repository contents

```
notebooks/      Colab notebooks, numbered in the order they were run (outputs preserved)
results/        CSV outputs: per-seed metrics, robustness grid, learning curves
figures/        Learning curves, heatmaps and result figures
models/         The 12 final trained agents (Stable-Baselines3 .zip files)
videos/         Demo videos used in the presentation
presentation/   Slide deck (.pptx) and speaker script
requirements.txt
```

## Which notebook produced what

Each training run was executed as its own Colab session, so the code lives in several notebooks. All of them share the same setup cells (installs, configuration, `DynamicsShiftWrapper`, `deterministic_evaluate`, `ProjectEvalCallback`, `build_model`); they differ in the training cell. Each notebook starts with a note saying exactly which cells ran.

| Notebook | What it ran | Hardware |
|---|---|---|
| `01_train_ppo_all_6_runs.ipynb` | PPO nominal seeds 0–2 and PPO randomized seeds 0–2 (Cell 6) | Colab CPU |
| `02_train_sac_nominal_seed0_to_300k.ipynb` | SAC nominal seed 0, first session to the 300K checkpoint | Colab CPU |
| `03_resume_sac_nominal_seed0_300k_to_500k.ipynb` | SAC nominal seed 0, resumed 300K → 500K | Colab CPU |
| `04_train_sac_nominal_seed1.ipynb` | SAC nominal seed 1 | NVIDIA L4 GPU |
| `05_train_sac_nominal_seed2_and_randomized_seed0.ipynb` | SAC nominal seed 2; SAC randomized seed 0 | NVIDIA L4 GPU |
| `06_train_sac_randomized_seed1_seed2_and_resume.ipynb` | SAC randomized seed 1; SAC randomized seed 2 (to 300K checkpoint, then resumed to 500K in Cell 9R) | NVIDIA L4 GPU |
| `07_evaluate_robustness_and_figures.ipynb` | Loads all 12 models; 5 × 5 robustness grid (Cell 12); tables, heatmaps, learning curves, figures (Cells 13–17); demo telemetry (Cell 18D) | NVIDIA L4 GPU |
| `08_make_demo_video.ipynb` | Demo videos for the presentation (drawn from MuJoCo state, no OpenGL) | Any Colab runtime |

## Experimental setup

| Setting | Value |
|---|---|
| Environment | Gymnasium Hopper-v5 (MuJoCo), built-in reward |
| Conditions | {PPO, SAC} × {nominal, randomized}, 3 seeds each = 12 agents |
| Training budget | 500,000 environment steps per run (PPO: 501,760 because it collects whole 2,048-step rollouts) |
| Randomized training | Mass × U[0.8, 1.2], friction × U[0.7, 1.3], resampled every episode; not observed by the policy |
| Validation | Every 25,000 steps, 10 deterministic nominal episodes |
| Robustness test | Final policy; mass scales {0.6, 0.8, 1.0, 1.2, 1.4} × friction scales {0.5, 0.75, 1.0, 1.25, 1.5}; 10 deterministic episodes per cell; identical evaluation seeds for every agent |
| Success | Survive all 1,000 steps and move at least 3 m forward |
| PPO | 2 × 256 tanh MLP, lr 3e-4, γ 0.99, n_steps 2,048, 10 epochs, batch 64, clip 0.2, GAE λ 0.95 |
| SAC | 2 × 256 ReLU MLP, lr 3e-4, γ 0.99, buffer 1M, batch 256, τ 0.005, automatic entropy, 10K random start steps |

## How to run

The notebooks are written for **Google Colab** with Google Drive for storage.

1. Upload the notebooks to Colab. In each one, `RESULTS_DIR` points to `/content/drive/MyDrive/RL_Part2/rl_part2_final_results`; change it if your Drive folder differs.
2. Run the setup cells (Cell 0 through the `build_model` cell), then the training cell listed in the table above. Each run takes roughly 1.5–2 hours on an L4 GPU (SAC) or CPU (PPO). Training cells refuse to overwrite a finished model.
3. After all 12 models exist in `models/`, run `07_evaluate_robustness_and_figures.ipynb` from Cell 10 onward to reproduce the grid evaluation, CSVs and figures.
4. To regenerate the demo videos, open `08_make_demo_video.ipynb` on a fresh runtime and choose **Runtime → Run all**.

To only inspect the results, no training is needed: load any model with `PPO.load("models/ppo_randomized_seed1.zip")` or `SAC.load(...)` and evaluate it with the wrapper from the setup cells.

Install the dependencies locally with:

```
pip install -r requirements.txt
```

## Key results

| Configuration | Nominal return (IQM) | Nominal success (IQM) | Grid return (IQM) | OOD return (IQM) |
|---|---|---|---|---|
| PPO Nominal | 2,979 | 83% | 969 | 700 |
| PPO Randomized | 2,817 | 70% | 1,193 | 956 |
| SAC Nominal | 3,281 | 100% | 1,032 | 708 |
| SAC Randomized | 2,930 | 70% | 1,061 | 826 |

- **H1 (SAC more sample-efficient): directionally supported.** Learning-curve area 2,293 vs. 1,871 on SAC's two complete seeds.
- **H2 (nominal policies brittle out of distribution): supported.** Robustness ratio 0.28 (PPO) and 0.22 (SAC), far below 0.70.
- **H3 (randomization improves robustness at < 10% nominal cost): partly supported.** Robustness rose for both algorithms; nominal cost was 5.5% for PPO and 10.7% for SAC.
- Results are strongly seed-dependent: PPO Randomized seed 1 reached 51% grid success, while its other two seeds reached 3% and 8%. Every seed is reported in `results/`.
- At 1.5× friction, every configuration scores below 600 in every cell.

## Changes from the Part 1 plan

- 3 seeds per condition instead of 5 (runtime; Part 1's fallback rule).
- SAC runs mostly on an L4 GPU (SAC nominal seed 0 on CPU), so wall-clock time is not compared.
- No observation/reward normalization for PPO; both algorithms see raw observations.
- IQM with every seed shown, instead of bootstrap confidence intervals (not meaningful with n = 3).
- Demo video drawn from MuJoCo simulation state, because the Colab OpenGL renderer crashed.

## Known issues and limitations

- **Missing validation history:** SAC nominal seed 0 and SAC randomized seed 2 crashed and were resumed from 300K checkpoints (model, optimizer and replay buffer). Their validation points from 25K–300K were held in memory and lost; they are left as gaps, not reconstructed.
- **PPO final update:** PPO's "500K" validation runs inside its last rollout, before the final policy update. For PPO randomized seeds 0 and 2, that update changed nominal return substantially (1,056 → 3,074 and 3,284 → 1,193). The robustness grid evaluates the final saved models.
- **Physics simplification:** mass scaling does not scale rotational inertia; friction scaling changes all three MuJoCo friction coefficients.
- **GPU nondeterminism:** SAC reruns on GPU will differ slightly even with the same seeds.
- **Library versions:** demo videos generated later with newer library versions differ from the original telemetry by < 0.1% at nominal physics.

## Files not included

Replay buffers (≈ 200 MB each) and intermediate checkpoints are excluded because of size. They are available on request.

## AI disclosure

Claude (Anthropic) was used to review the notebooks and results, recompute per-seed statistics from the CSV outputs, diagnose the Colab rendering failure and write the demo-video notebook, draft the presentation and speaker notes, and organize this repository. All experiments were designed and run by the author, who checked the reported numbers and is responsible for the conclusions.

## References

See the final slide of `presentation/RL_Part2_Final_Presentation.pptx`. Main sources: Schulman et al. (2017) PPO; Haarnoja et al. (2018) SAC; Tobin et al. (2017) and Peng et al. (2018) domain randomization; Rajeswaran et al. (2017) EPOpt; Henderson et al. (2018); Agarwal et al. (2021); Raffin et al. (2021) Stable-Baselines3; Todorov et al. (2012) MuJoCo.
