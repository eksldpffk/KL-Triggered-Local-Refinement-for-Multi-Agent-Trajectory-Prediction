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

## Core idea

The method uses KL divergence as a correction-necessity score.
A fast probabilistic forecaster first predicts future trajectories and uncertainty for all valid agents. Nearby pairs pass through a cheap distance gate. For each candidate pair, the method builds a minimally safety-adjusted distribution q<sub>safe</sub> and compares it with the original forecast p<sub>fast</sub>.

If the KL divergence is high, the original prediction would need a meaningful safety correction, so only that local pair is sent to the heavier refiner.

## Method

<img src= "assets/KL_arc.png" align="right" width="500">

1. **Fast probabilistic forecasting**<br>
   A lightweight model predicts a Gaussian future trajectory distribution for each agent: `p_fast = N(μ, σ²)`.
2. **Distance pre-filter**<br>
   Pairs that stay far apart are filtered out using a simple distance check based on `d_min + safety_margin`.
3. **Safety-adjusted local distribution**<br>
   For candidate pairs that cross the hard separation boundary `d_min`, a minimally corrected local distribution `q_safe` is constructed.
4. **Pairwise KL trigger**<br>
   The method computes `KL(q_safe || p_fast)`.<br>
   High KL means that the predicted interaction strongly disagrees with the local safety correction.
5. **Contextual threshold and top-k selection**<br>
   The trigger threshold adapts to scene density, predictive uncertainty, and the requested safety level. Only the highest-risk pairs are selected.
6. **Local refinement**<br>
   The refiner corrects only selected interacting pairs. Other agents keep the original fast forecast.
8. **Safety distillation**<br>
   During training, the refined trajectories of risky pairs are used as extra targets, while the model still learns from the original ground-truth trajectories.


   
## Results

Evaluation on the Argoverse 2 validation set.

| Method | ADE ↓ | FDE ↓ | Approx. Collision ↓ | Separation Violation ↓ | P50 Latency ↓ | P95 Latency ↓ | Refine Rate ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fast only | **0.819** | 2.001 | 19.0% | 60.5% | **1.34 ms** | **2.03 ms** | **0.0%** |
| Always refine | 0.880 | 2.044 | **0.0%** | **32.9%** | 358.45 ms | 2052.30 ms | 100.0% |
| Scene-level switching | 0.875 | 2.041 | 7.3% | 48.4% | 4.68 ms | 2028.25 ms | 34.3% |
| **KL-triggered local (Ours)** | 0.866 | 2.033 | 16.0% | 56.3% | 4.66 ms | 36.20 ms | 34.3% |
| **Ours + Safety Distillation** | **0.838** | **1.931** | 16.0% | 57.3% | **4.52 ms** | **35.65 ms** | **32.7%** |

The results show the expected safety–computation trade-off.

**Always refine** achieves the best safety, but it is extremely expensive because the whole scene is corrected every time.  
**Fast only** is the fastest method, but it has the highest collision rate.

Our **KL-triggered local refinement** keeps the refinement rate similar to scene-level switching, but reduces latency from **2028 ms to 36 ms** by correcting only selected risky interactions. The trade-off is weaker safety improvement because only a small part of the scene is refined.

Adding **Safety Distillation** improves prediction quality, reducing ADE from **0.866 to 0.838** and FDE from **2.033 to 1.931**, while also reducing refinement usage from **34.3% to 32.7%**. Approximate collision rate remains unchanged.

Overall, the method does not maximize safety alone. Its advantage is a much better **safety–latency trade-off**, avoiding expensive full-scene refinement while still improving safety over the fast-only model.
