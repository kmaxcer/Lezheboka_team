import importlib.util
for m in ['catboost','xgboost','lightgbm']: print(m,importlib.util.find_spec(m))
