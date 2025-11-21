import pandas as pd
from src.api.paysim_api import build_feature_row, TransactionInput


def test_feature_engineering_output_shape():
    tx = TransactionInput(
        amount=10000,
        type="CASH_OUT",
        oldbalanceOrg=12000,
        newbalanceOrig=2000,
        oldbalanceDest=0,
        newbalanceDest=10000,
        step=50,
    )

    X = build_feature_row(tx)

    assert isinstance(X, pd.DataFrame), "Output must be a DataFrame"
    assert X.shape[0] == 1, "Should return exactly one row"
    assert X.shape[1] > 10, "Engineered features + type encodings expected"
