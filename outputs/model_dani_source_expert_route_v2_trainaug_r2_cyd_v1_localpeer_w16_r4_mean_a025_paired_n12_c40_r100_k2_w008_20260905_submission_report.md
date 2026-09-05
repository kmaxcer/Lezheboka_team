# Actual-gap paired-AOI overlay on w16/r4/mean a=.25 base

Formula: base=model_dani_source_expert_route_v2_trainaug_r2_cyd_v1_localpeer_w16_r4_mean_a025_submission.csv; paired correction: if finite private-only n12_c40_r100_k2 peer, pred=clip(0.92*base+0.08*peer,[-0.2,1.1]), else base
Rows: 3112; paired coverage: 0.753856; SHA256: `69a525e610e0d6a6a2bfe6d404374cea58ad7cbaf82a9c5e2b4e2d75efccd21b`.

Actual peer coverage for requested configurations:
         config    n  peer_n  coverage
n12_c40_r100_k2 3112    2346  0.753856
n12_c80_r100_k2 3112    1886  0.606041
n16_c60_r125_k2 3112    2371  0.761889

Leakage safety: peer maps use train + visible private rows; synthetic gaps are query-only. Four-mask audit is in research/paired_aoi_trainaug_local_audit_v1_report.md and research/paired_aoi_v2_private_only_trainaug_localpeer_multi_report_20260905.md.
No submission/upload performed.
