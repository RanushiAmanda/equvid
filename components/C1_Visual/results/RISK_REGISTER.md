# C1 (Visual) risk register

Status at PP1, 22 October 2026. Likelihood and impact are my own judgement (H/M/L).

| # | Risk | L / I | Evidence so far | Mitigation | Status |
|---|---|---|---|---|---|
| 1 | The score rests on one corpus (AVA); MELD checks only the false-positive side | H / H | MELD control passes on presence; its framing gaps shrink from dev to train | AMI as framing contrast corpus; report intervals, never a bare point estimate | AMI not started |
| 2 | AVA gender labels are predicted, not annotated; model over-calls female | H / H | 89.0% accurate where decided; 18.2% of "female" labels wrong vs 4.3% of "male"; 29% refused, non-randomly | Blind validation done; intervals reported; decide with supervisor between hand annotation plus kappa and validated model labels | Open |
| 3 | AVA face boxes come from the authors' automatic detector; recall by gender unmeasured | M / H | Roth et al.: annotators discarded false-positive tracks only | Hand count of visible against annotated faces, by gender, on a frame sample | Not started |
| 4 | Small validation sample | H / M | 160 usable labels, 20 of 88 videos, one annotator | State as limit; second annotator and kappa; larger sample | Open |
| 5 | Centrality, close-up and solo anchors are conventions, not measurements | H / M | Only the presence anchor is derived (Annenberg 2.30:1) | Report sign and interval, not magnitude; ground anchors in the group perception study | Open |
| 6 | Pooling rule is undecided; solo is fragile (2.11, interval 0.8 to 3.4) | M / M | Tiers unchanged across rules, magnitudes move (solo up to 4.47) | Decide one rule with Temporal; read solo by sign | Open |
| 7 | Divergence from the proposal (six features to four, closed form instead of Ridge, predicted labels, sign) | M / M | Each change has measured evidence in `RESULTS.md` | Log each change; inform supervisors; keep Ridge as a later comparison | Supervisor note to be sent |
| 8 | Shared augmentation protocol and calibration set not frozen; AVA is NA for C2 (Decision 6), so a fused C1+C2 score has no AVA rows | H / H | Shared conventions note of 6 Oct | Raise with the group; supply per-person AVA labels with source and confidence | Open, group decision |
| 9 | MELD cannot separate gender from character (3 women, 3 men); single face assumed to be the speaker | H / M | Best case p = 0.10 under relabelling; spread between characters exceeds the gender gap | Treat MELD as a control; spot-check the speaker assumption | Open |
| 10 | Schedule: roughly 40% of the proposal's work packages complete at PP1 | M / H | AMI, detector audit, augmentation and regression, fusion hand-off still to do | Order: detector audit, AMI, labels, MELD test split, augmentation and regression | In progress |
