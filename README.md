# KL-Triggered-Local-Refinement-for-Multi-Agent-Trajectory-Prediction
A multi-agent trajectory prediction framework that uses KL divergence to detect risky interactions and refine only the agents that need correction.

## Why this project?

- Fast trajectory predictors are useful for real-time systems, but they may miss important local interactions between nearby agents.
- Running a heavy safety refinement step on the entire scene can reduce risk, but it also adds unnecessary computation when most agents are already behaving safely.

This project explores a simple idea: **refine only the interactions that actually need it.**

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

![Architecture](assets/architecture.png)
                         ↓
                  Safety distillation
                  during training
