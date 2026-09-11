# KL-Triggered Local Refinement for Multi-Agent Motion Forecasting

A selective-compute framework for dense driving scenes.
The goal is to keep a fast probabilistic forecaster for most interactions and spend extra computation only on local agent pairs whose predictions strongly conflict with a safety-adjusted alternative.

## Problem

Multi-agent motion forecasting has a practical trade-off:
- A lightweight forecaster is fast enough for real-time use, but it can produce locally inconsistent or unsafe interactions.
- Refining the entire scene can improve safety, but it wastes computation when only one or two pairs are problematic.
- Distance alone is not enough to decide which close interactions require correction: the same geometric conflict can be more or less significant depending on predictive uncertainty.

The project asks:
> Can we identify only the interactions that truly need correction and refine them locally, instead of refining the whole scene?

## How it works

The framework follows four main steps:
1. **Fast prediction**  
   A lightweight probabilistic planner predicts future trajectories and uncertainty for all agents.
2. **KL-based risk detection**  
   Nearby agent pairs are compared with a minimally safety-adjusted version of their predicted trajectories.  
   A high KL divergence means that the original prediction strongly conflicts with the safer alternative.
3. **Local refinement**  
   Only high-risk interactions are passed to a heavier correction module. The rest of the scene keeps the original fast prediction.
4. **Safety distillation**  
   During training, refined local trajectories can be used as teacher targets so that the fast planner gradually learns safer behaviour.

The KL threshold can also adapt to scene context, including traffic density, predictive uncertainty, and the desired safety level.

## Architecture
<p align="center">
   <img src= "assets/kl_architecter.png">
</p>
