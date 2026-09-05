# New test HGB baseline

{
  "dataset": "data_update_20260905_1350",
  "test_rows": 49190,
  "test_gap_rows": 2323,
  "test_aoi": 20,
  "date_min": "2010-04-01",
  "date_max": "2024-10-30",
  "label_provenance": "new test gap labels never read; old private released ground truth joined only as training target",
  "outputs": [
    {
      "candidate": "outputs\\test_20260905_1350\\model_newtest_extended_hgb_regular_20260905.csv",
      "formula": "HistGradientBoosting on old train + old private with released old-gap labels, 3x18% AOI-year pseudo-mask OOF blocks; feature_hgb_v2 build_features + extra_features; query=new test is_synthetic_gap rows",
      "rows": 2323,
      "finite": true,
      "unique_keys": 2323,
      "feature_count": 69,
      "train_rows": 34731,
      "released_old_labels": 20753,
      "new_test_gap_rows": 2323,
      "test_sha256": "f7ba087818175ea644fc9fe652c6a3874b5bfcde7df197d3e7541cb95876f855",
      "train_sha256": "a75e530d0fb51581ad6800f84b3875233778801491f02236917862faf9b424ec",
      "private_sha256": "3c5c0e27eef8266bcf6dce09c9b556c073cee3902c065a94e4ea7a59edb00993",
      "old_truth_sha256": "50d694a92187b7e8a2fca8a2b72458d9a8042726bd9d85634eb7a85fa5174088",
      "candidate_sha256": "fd14dac6f587c89c8214a925f134bd1f92498174dec376001abf608ae97d5456",
      "no_upload": true,
      "production_baseline_overwritten": false,
      "seconds": 144.4
    },
    {
      "candidate": "outputs\\test_20260905_1350\\model_newtest_extended_hgb_wide_20260905.csv",
      "formula": "HistGradientBoosting on old train + old private with released old-gap labels, 3x18% AOI-year pseudo-mask OOF blocks; feature_hgb_v2 build_features + extra_features; query=new test is_synthetic_gap rows",
      "rows": 2323,
      "finite": true,
      "unique_keys": 2323,
      "feature_count": 69,
      "train_rows": 34731,
      "released_old_labels": 20753,
      "new_test_gap_rows": 2323,
      "test_sha256": "f7ba087818175ea644fc9fe652c6a3874b5bfcde7df197d3e7541cb95876f855",
      "train_sha256": "a75e530d0fb51581ad6800f84b3875233778801491f02236917862faf9b424ec",
      "private_sha256": "3c5c0e27eef8266bcf6dce09c9b556c073cee3902c065a94e4ea7a59edb00993",
      "old_truth_sha256": "50d694a92187b7e8a2fca8a2b72458d9a8042726bd9d85634eb7a85fa5174088",
      "candidate_sha256": "65cde19191bc2a74afa01fe4d5db94ac25d1016838a3cffc8c5a7ba4576a9fc7",
      "no_upload": true,
      "production_baseline_overwritten": false,
      "seconds": 163.3
    }
  ]
}
Meanblend candidate: model_newtest_extended_hgb_meanblend_20260905.csv, formula 0.5 regular + 0.5 wide clipped [-0.2,1.1], rows 2323, SHA256 d33b28fb7f7fc5bb79f2a5dc8e49d6da3cdb0e7aca3fc627ccc4d028abf1a57e, finite/unique contract passed, no upload.
