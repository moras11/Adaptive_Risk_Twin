import lightgbm as lgb


def test_model_loading():
    try:
        model = lgb.Booster(model_file="models/lgbm_fraud.txt")
    except Exception as e:
        assert False, f"Model failed to load: {e}"

    assert True
