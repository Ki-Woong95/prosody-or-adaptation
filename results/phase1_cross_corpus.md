# Phase-1 cross-corpus teacher-target evaluation

The alignment-corrected CASPER Phase-1 best checkpoint was frozen and evaluated
without retraining. Teacher targets were newly extracted from the complete
Buckeye, Switchboard, and AMI-IHM validation splits using the same CREPE-tiny
and canonical-grid implementation used for CASPER v2. Target normalization was
fixed to the CASPER training statistics.

| Validation corpus | Utterances | Total loss | F0 cents MAE | Voicing P / R / F1 | Delta-F0 RMSE | Energy RMSE / r | Tilt RMSE / r |
|---|---:|---:|---:|---:|---:|---:|---:|
| CASPER | 7,824 | 0.768 | 146.57 | 0.876 / 0.905 / 0.890 | 0.0118 | 0.238 / 0.998 | 0.162 / 0.998 |
| Buckeye | 797 | 1.006 | 194.61 | 0.878 / 0.887 / 0.882 | 0.0126 | 0.267 / 0.996 | 0.176 / 0.997 |
| Switchboard | 20,601 | 0.963 | 184.38 | 0.820 / 0.913 / 0.864 | 0.0114 | 0.297 / 0.993 | 0.179 / 0.995 |
| AMI-IHM | 9,428 | 0.684 | 56.33 | 0.806 / 0.971 / 0.880 | 0.0097 | 0.341 / 0.988 | 0.238 / 0.996 |

Valid frame counts were 5,825,568 for CASPER, 782,284 for Buckeye, 4,562,841
for Switchboard, and 1,486,765 for AMI-IHM. The complete cross-corpus run took
1,190.8 seconds on the local GPU.

These values measure agreement with the distillation teacher, not independent
ground-truth prosody. F0 and delta-F0 metrics use the teacher-defined voiced and
voiced-transition masks. Losses use CASPER-train normalization, making the
reported loss scale consistent across the four validation sets.
