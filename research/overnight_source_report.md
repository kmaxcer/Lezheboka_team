# Overnight source-aware evaluator

Research-only run; `outputs/model_dani_tuned*` and input CSVs were not modified.

## Protocols

- exact_hidden_doy: private synthetic DOYs projected onto train 2019--2024;
- private_like: 15% random known private rows per AOI/year, seeds 0/1/2;
- private_2025_date: optional 2025 date-multiplicity stress proxy.

## Source routing

`soft` is posterior-weighted source interpolation (production logic); `hard` selects the modal source; `oracle_source` is evaluation-only.

         protocol              method  source    n   metric      mae
 exact_hidden_doy   lag_oracle_source     all 1114 0.061100 0.042359
 exact_hidden_doy            lag_soft landsat  326 0.062948 0.042885
 exact_hidden_doy           base_soft landsat  326 0.063353 0.043727
 exact_hidden_doy  base_oracle_source     all 1114 0.064741 0.044862
 exact_hidden_doy           base_hard landsat  326 0.065592 0.045792
 exact_hidden_doy            lag_hard landsat  326 0.065804 0.045079
 exact_hidden_doy            lag_soft     all 1114 0.067603 0.048193
 exact_hidden_doy            lag_soft   modis  343 0.069214 0.052948
 exact_hidden_doy            lag_soft      s2  445 0.069609 0.048415
 exact_hidden_doy           base_soft      s2  445 0.070514 0.049149
 exact_hidden_doy           base_soft     all 1114 0.070554 0.050600
 exact_hidden_doy            lag_hard     all 1114 0.070577 0.048967
 exact_hidden_doy            lag_hard      s2  445 0.071739 0.046961
 exact_hidden_doy           base_hard      s2  445 0.072561 0.048011
 exact_hidden_doy           base_hard     all 1114 0.072669 0.051258
 exact_hidden_doy            lag_hard   modis  343 0.073383 0.055263
 exact_hidden_doy           base_soft   modis  343 0.076822 0.059015
 exact_hidden_doy           base_hard   modis  343 0.078938 0.060666
 exact_hidden_doy base_route_accuracy     all 1114 0.809892 0.457619
 exact_hidden_doy  lag_route_accuracy     all 1114 0.809892 0.457619
private_2025_date            lag_soft   modis  326 0.068347 0.054078
private_2025_date   lag_oracle_source     all 2775 0.069705 0.039173
private_2025_date  base_oracle_source     all 2775 0.071639 0.041477
private_2025_date            lag_soft      s2 1808 0.071820 0.042826
private_2025_date            lag_hard      s2 1808 0.072205 0.042457
private_2025_date            lag_soft     all 2775 0.072601 0.042947
private_2025_date           base_soft      s2 1808 0.072899 0.044848
private_2025_date           base_hard      s2 1808 0.073041 0.044319
private_2025_date           base_soft     all 2775 0.074338 0.044908
private_2025_date            lag_hard     all 2775 0.074531 0.043900
private_2025_date           base_hard     all 2775 0.075640 0.045510
private_2025_date           base_soft   modis  326 0.076272 0.060850
private_2025_date            lag_soft landsat  641 0.076773 0.037629
private_2025_date            lag_hard   modis  326 0.076934 0.060710
private_2025_date           base_soft landsat  641 0.077303 0.036971
private_2025_date            lag_hard landsat  641 0.079577 0.039420
private_2025_date           base_hard landsat  641 0.079841 0.038708
private_2025_date           base_hard   modis  326 0.081118 0.065488
private_2025_date base_route_accuracy     all 2775 0.876789 0.310362
private_2025_date  lag_route_accuracy     all 2775 0.876789 0.310362

## Residual correction / low-rank diversity

Corrections are fit only from an additional observed-only calibration mask (10% of genuinely observed rows) and scored on the outer hidden rows. Date/crop residuals use robust median, count shrinkage and a +/-0.03 cap. PCA stacks are rank 1/2 and cross-fitted. The superseded parity diagnostic is saved separately as `overnight_correction_diagnostic_metrics.csv` and is not used for selection.

        protocol partition                method    n  cal_n     rmse      mae  baseline_rmse                fit_source
exact_hidden_doy exact2019     safe_loo_date_w10  249   3029 0.064945 0.047028       0.065045 observed_calibration_only
exact_hidden_doy exact2019     safe_loo_date_w20  249   3029 0.064861 0.046898       0.065045 observed_calibration_only
exact_hidden_doy exact2019 safe_loo_crop_doy_w10  249   3029 0.064953 0.047061       0.065045 observed_calibration_only
exact_hidden_doy exact2019 safe_loo_crop_doy_w20  249   3029 0.064874 0.046970       0.065045 observed_calibration_only
exact_hidden_doy exact2019      safe_loo_doy_w10  249   3029 0.064943 0.047110       0.065045 observed_calibration_only
exact_hidden_doy exact2019      safe_loo_doy_w20  249   3029 0.064853 0.047065       0.065045 observed_calibration_only
exact_hidden_doy exact2019   safe_pca_rank1_b0.5  249   3029 0.064434 0.046666       0.065045 observed_calibration_only
exact_hidden_doy exact2019   safe_pca_rank1_b1.0  249   3029 0.065483 0.047984       0.065045 observed_calibration_only
exact_hidden_doy exact2019   safe_pca_rank2_b0.5  249   3029 0.064493 0.046168       0.065045 observed_calibration_only
exact_hidden_doy exact2019   safe_pca_rank2_b1.0  249   3029 0.065540 0.047547       0.065045 observed_calibration_only
exact_hidden_doy exact2020     safe_loo_date_w10  222   3031 0.064183 0.046670       0.064274 observed_calibration_only
exact_hidden_doy exact2020     safe_loo_date_w20  222   3031 0.064095 0.046563       0.064274 observed_calibration_only
exact_hidden_doy exact2020 safe_loo_crop_doy_w10  222   3031 0.064049 0.046656       0.064274 observed_calibration_only
exact_hidden_doy exact2020 safe_loo_crop_doy_w20  222   3031 0.063841 0.046557       0.064274 observed_calibration_only
exact_hidden_doy exact2020      safe_loo_doy_w10  222   3031 0.064162 0.046696       0.064274 observed_calibration_only
exact_hidden_doy exact2020      safe_loo_doy_w20  222   3031 0.064061 0.046612       0.064274 observed_calibration_only
exact_hidden_doy exact2020   safe_pca_rank1_b0.5  222   3031 0.062106 0.043649       0.064274 observed_calibration_only
exact_hidden_doy exact2020   safe_pca_rank1_b1.0  222   3031 0.062950 0.043412       0.064274 observed_calibration_only
exact_hidden_doy exact2020   safe_pca_rank2_b0.5  222   3031 0.062083 0.043620       0.064274 observed_calibration_only
exact_hidden_doy exact2020   safe_pca_rank2_b1.0  222   3031 0.062982 0.043374       0.064274 observed_calibration_only
exact_hidden_doy exact2021     safe_loo_date_w10  130   3043 0.089806 0.066833       0.090260 observed_calibration_only
exact_hidden_doy exact2021     safe_loo_date_w20  130   3043 0.089366 0.066535       0.090260 observed_calibration_only
exact_hidden_doy exact2021 safe_loo_crop_doy_w10  130   3043 0.090047 0.066972       0.090260 observed_calibration_only
exact_hidden_doy exact2021 safe_loo_crop_doy_w20  130   3043 0.089843 0.066806       0.090260 observed_calibration_only
exact_hidden_doy exact2021      safe_loo_doy_w10  130   3043 0.090071 0.066982       0.090260 observed_calibration_only
exact_hidden_doy exact2021      safe_loo_doy_w20  130   3043 0.089891 0.066816       0.090260 observed_calibration_only
exact_hidden_doy exact2021   safe_pca_rank1_b0.5  130   3043 0.090390 0.066544       0.090260 observed_calibration_only
exact_hidden_doy exact2021   safe_pca_rank1_b1.0  130   3043 0.092260 0.067770       0.090260 observed_calibration_only
exact_hidden_doy exact2021   safe_pca_rank2_b0.5  130   3043 0.090512 0.067129       0.090260 observed_calibration_only
exact_hidden_doy exact2021   safe_pca_rank2_b1.0  130   3043 0.092847 0.068738       0.090260 observed_calibration_only
exact_hidden_doy exact2022     safe_loo_date_w10  192   3039 0.072263 0.053016       0.072241 observed_calibration_only
exact_hidden_doy exact2022     safe_loo_date_w20  192   3039 0.072294 0.053022       0.072241 observed_calibration_only
exact_hidden_doy exact2022 safe_loo_crop_doy_w10  192   3039 0.072302 0.053018       0.072241 observed_calibration_only
exact_hidden_doy exact2022 safe_loo_crop_doy_w20  192   3039 0.072381 0.053001       0.072241 observed_calibration_only
exact_hidden_doy exact2022      safe_loo_doy_w10  192   3039 0.072268 0.053006       0.072241 observed_calibration_only
exact_hidden_doy exact2022      safe_loo_doy_w20  192   3039 0.072307 0.052980       0.072241 observed_calibration_only
exact_hidden_doy exact2022   safe_pca_rank1_b0.5  192   3039 0.070261 0.051379       0.072241 observed_calibration_only
exact_hidden_doy exact2022   safe_pca_rank1_b1.0  192   3039 0.069867 0.050884       0.072241 observed_calibration_only
exact_hidden_doy exact2022   safe_pca_rank2_b0.5  192   3039 0.070187 0.050969       0.072241 observed_calibration_only
exact_hidden_doy exact2022   safe_pca_rank2_b1.0  192   3039 0.069857 0.049871       0.072241 observed_calibration_only
exact_hidden_doy exact2023     safe_loo_date_w10  169   3034 0.076063 0.052357       0.076162 observed_calibration_only
exact_hidden_doy exact2023     safe_loo_date_w20  169   3034 0.075973 0.052323       0.076162 observed_calibration_only
exact_hidden_doy exact2023 safe_loo_crop_doy_w10  169   3034 0.076250 0.052389       0.076162 observed_calibration_only
exact_hidden_doy exact2023 safe_loo_crop_doy_w20  169   3034 0.076354 0.052365       0.076162 observed_calibration_only
exact_hidden_doy exact2023      safe_loo_doy_w10  169   3034 0.076237 0.052393       0.076162 observed_calibration_only
exact_hidden_doy exact2023      safe_loo_doy_w20  169   3034 0.076322 0.052373       0.076162 observed_calibration_only
exact_hidden_doy exact2023   safe_pca_rank1_b0.5  169   3034 0.075305 0.051901       0.076162 observed_calibration_only
exact_hidden_doy exact2023   safe_pca_rank1_b1.0  169   3034 0.076328 0.053039       0.076162 observed_calibration_only
exact_hidden_doy exact2023   safe_pca_rank2_b0.5  169   3034 0.074824 0.051476       0.076162 observed_calibration_only
exact_hidden_doy exact2023   safe_pca_rank2_b1.0  169   3034 0.075354 0.052041       0.076162 observed_calibration_only
exact_hidden_doy exact2024     safe_loo_date_w10  152   3040 0.059201 0.042454       0.059232 observed_calibration_only
exact_hidden_doy exact2024     safe_loo_date_w20  152   3040 0.059181 0.042355       0.059232 observed_calibration_only
exact_hidden_doy exact2024 safe_loo_crop_doy_w10  152   3040 0.059195 0.042433       0.059232 observed_calibration_only
exact_hidden_doy exact2024 safe_loo_crop_doy_w20  152   3040 0.059177 0.042314       0.059232 observed_calibration_only
exact_hidden_doy exact2024      safe_loo_doy_w10  152   3040 0.059289 0.042519       0.059232 observed_calibration_only
exact_hidden_doy exact2024      safe_loo_doy_w20  152   3040 0.059360 0.042484       0.059232 observed_calibration_only
exact_hidden_doy exact2024   safe_pca_rank1_b0.5  152   3040 0.057522 0.039236       0.059232 observed_calibration_only
exact_hidden_doy exact2024   safe_pca_rank1_b1.0  152   3040 0.058664 0.038909       0.059232 observed_calibration_only
exact_hidden_doy exact2024   safe_pca_rank2_b0.5  152   3040 0.058225 0.039758       0.059232 observed_calibration_only
exact_hidden_doy exact2024   safe_pca_rank2_b1.0  152   3040 0.060285 0.039756       0.059232 observed_calibration_only
    private_like   random0     safe_loo_date_w10 2644   1495 0.081143 0.051369       0.081177 observed_calibration_only
    private_like   random0     safe_loo_date_w20 2644   1495 0.081113 0.051342       0.081177 observed_calibration_only
    private_like   random0 safe_loo_crop_doy_w10 2644   1495 0.081191 0.051402       0.081177 observed_calibration_only
    private_like   random0 safe_loo_crop_doy_w20 2644   1495 0.081217 0.051414       0.081177 observed_calibration_only
    private_like   random0      safe_loo_doy_w10 2644   1495 0.081199 0.051407       0.081177 observed_calibration_only
    private_like   random0      safe_loo_doy_w20 2644   1495 0.081234 0.051424       0.081177 observed_calibration_only
    private_like   random0   safe_pca_rank1_b0.5 2644   1495 0.080228 0.050698       0.081177 observed_calibration_only
    private_like   random0   safe_pca_rank1_b1.0 2644   1495 0.081298 0.052110       0.081177 observed_calibration_only
    private_like   random0   safe_pca_rank2_b0.5 2644   1495 0.079890 0.050261       0.081177 observed_calibration_only
    private_like   random0   safe_pca_rank2_b1.0 2644   1495 0.080530 0.051015       0.081177 observed_calibration_only
    private_like   random1     safe_loo_date_w10 2644   1495 0.075927 0.051375       0.075948 observed_calibration_only
    private_like   random1     safe_loo_date_w20 2644   1495 0.075910 0.051346       0.075948 observed_calibration_only
    private_like   random1 safe_loo_crop_doy_w10 2644   1495 0.075871 0.051347       0.075948 observed_calibration_only
    private_like   random1 safe_loo_crop_doy_w20 2644   1495 0.075810 0.051304       0.075948 observed_calibration_only
    private_like   random1      safe_loo_doy_w10 2644   1495 0.075852 0.051322       0.075948 observed_calibration_only
    private_like   random1      safe_loo_doy_w20 2644   1495 0.075771 0.051253       0.075948 observed_calibration_only
    private_like   random1   safe_pca_rank1_b0.5 2644   1495 0.074094 0.049895       0.075948 observed_calibration_only
    private_like   random1   safe_pca_rank1_b1.0 2644   1495 0.074759 0.050765       0.075948 observed_calibration_only
    private_like   random1   safe_pca_rank2_b0.5 2644   1495 0.074014 0.049523       0.075948 observed_calibration_only
    private_like   random1   safe_pca_rank2_b1.0 2644   1495 0.074283 0.049529       0.075948 observed_calibration_only

Safe PCA stacks improve the local soft baseline, but they are not a production replacement: on the same exact hidden-DOY rows the current Dani HGB+lag 80/20 blend is RMSE 0.062606, versus 0.069389 for safe PCA rank-2/blend-0.5. On the random private-like protocol the blend is 0.069449, versus 0.076945 for safe PCA rank-2/blend-0.5. Mixing PCA into the production blend was also tested separately and did not improve it. Therefore no `overnight_pca2_submission_candidate.csv` is materialized or recommended.

The observed-only safe PCA gains are retained as diagnostics (local-soft baseline: exact 0.070554 -> 0.069389; private-like 0.078576 -> 0.076945), not as a final submission candidate.

## Production baseline on identical rows

| protocol | HGB | lag component | HGB+lag 80/20 |
|---|---:|---:|---:|
| exact_hidden_doy (1114) | 0.063406 | 0.067603 | **0.062606** |
| private_like (7932) | 0.069826 | 0.076287 | **0.069449** |

Source/year slices and MAE are in `overnight_baseline_aggregate.csv`.

Artifacts: `overnight_source_metrics.csv`, `overnight_source_aggregate.csv`, `overnight_correction_metrics.csv`, `overnight_correction_aggregate.csv`, `overnight_correction_diagnostic_metrics.csv`, `overnight_baseline_metrics.csv`, `overnight_baseline_aggregate.csv`, `overnight_baseline_compare.py`, `overnight_source_eval.py`.
