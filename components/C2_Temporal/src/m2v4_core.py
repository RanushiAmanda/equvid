"""M2 v0.4 core: participation fairness from REAL data only (no synthetic labels, no training).
 + = women favoured, - = men favoured, 0 = fair by definition (F and M equal)."""
import numpy as np

LN4 = float(np.log(4.0))


def c10(x, anchor):
    """map x to -10..+10; +-anchor (or more) = +-10."""
    return 10.0 * float(np.clip(x / anchor, -1.0, 1.0))


def components(f, f6, spec):
    """f: dict from shard_features; f6: float or nan. Returns the 4 lines and the fairness score."""
    g = f6 if f6 == f6 else f['log_parity_time']                       # time per screen-second, else per person
    A = c10(g, LN4)                                                      # speaking time vs opportunity
    B = c10(f['log_len_ratio'], LN4)   # turn length (turn count only shown, see note: more turns = fragmented, not favour)
    C = -c10(f['interruption_asym'], spec['scale_f4'])                  # being cut more = disadvantage
    gap = np.nan_to_num(abs(f['position_gap'])) / spec['scale_gap']
    sdr = np.nan_to_num(abs(f['pos_sd_ratio'])) / spec['scale_sd']
    T = 10.0 * float(min(1.0, 0.5 * (gap + sdr)))                       # timing imbalance, size only (no direction)
    return {'A': A, 'B': B, 'C': C, 'T': T, 'fair': (A + B + C) / 3.0, 'mode': 'P' if f6 == f6 else 'N'}
