# HGB architecture holdout

{
  "protocol": "old private synthetic gaps held out; released labels excluded from training/context; old train + old private visible only; 3x18% AOI-year pseudo-mask blocks",
  "metrics": [
    {
      "model": "regular",
      "n": 3112,
      "rmse": NaN,
      "mae": NaN,
      "pred_mean": 0.37888538753608475
    },
    {
      "model": "wide",
      "n": 3112,
      "rmse": NaN,
      "mae": NaN,
      "pred_mean": 0.37856524007821474
    }
  ],
  "seconds": 108.8,
  "no_upload": true
}
Status: invalid attempt; released GT was not joined into truth before scoring, so NaN metrics are discarded.
