# Calibrated HGB v3 audit (2026-09-05)

Исправлена потенциальная ошибка в `extra_features`: коэффициенты `numpy.polyfit` применены как `intercept + slope * sensor`.

Year holdout (6 leakage-safe folds, 1,114 rows): wide RMSE 0.0626482575, regular 0.0627341055, default 0.0630137649. Значения совпали с сохранённым v2 экспериментом до числовой точности, поэтому исправление не дало самостоятельного выигрыша.

New-test candidates materialized separately in `outputs/test_20260905_1350_calfix/`; 2,323 rows each, required columns, unique keys, finite values. New hidden labels were never read. Upload/submission not performed.
