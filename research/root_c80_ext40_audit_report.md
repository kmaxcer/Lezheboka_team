# c80 peer + ext40 audit

Three leakage-safe private-like masks (0, 1, 70404); c80 maps use only visible same-year/same-date targets. Hidden labels are used only for scoring.

                    candidate    n  pooled_rmse  pooled_baseline_rmse  pooled_delta_rmse  worst_seed_delta  all_seed_improve
c80_hist_0.09_shock10_state05 7932     0.068536               0.06871          -0.000174         -0.000080              True
c80_hist_0.10_shock10_state05 7932     0.068537               0.06871          -0.000173         -0.000065              True
                c80_hist_0.09 7932     0.068548               0.06871          -0.000162         -0.000052              True
                c80_hist_0.10 7932     0.068549               0.06871          -0.000161         -0.000037              True
c80_hist_0.12_shock10_state05 7932     0.068551               0.06871          -0.000159         -0.000020              True
                c80_hist_0.07 7932     0.068557               0.06871          -0.000152         -0.000071              True
                c60_hist_0.07 7932     0.068573               0.06871          -0.000137         -0.000036              True
                        ext40 7932     0.068710               0.06871           0.000000          0.000000              True
                c80_hist_0.12 7932     0.068563               0.06871          -0.000147          0.000007             False
                c60_hist_0.09 7932     0.068575               0.06871          -0.000134          0.000003             False
                c60_hist_0.10 7932     0.068584               0.06871          -0.000126          0.000030             False
                c80_hist_0.15 7932     0.068612               0.06871          -0.000098          0.000105             False
                c60_hist_0.12 7932     0.068615               0.06871          -0.000095          0.000100             False
                c60_hist_0.15 7932     0.068697               0.06871          -0.000013          0.000247             False

The selected rules are history/non-canonical only; 2025 rows remain the ext40 anchor.
No old output was overwritten.
