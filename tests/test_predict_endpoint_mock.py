from src.api.paysim_api import predict_fraud, TransactionInput
from pydantic import BaseModel


def test_predict_fraud_mock():
    tx = TransactionInput(
        amount=5000,
        type="TRANSFER",
        oldbalanceOrg=8000,
        newbalanceOrig=3000,
        oldbalanceDest=0,
        newbalanceDest=5000,
        step=150,
    )

    result = predict_fraud(tx)

    # Convert Pydantic model → dict
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    elif hasattr(result, "dict"):
        result = result.dict()

    assert "fraud_probability" in result, "Missing fraud probability"
    assert "is_fraud" in result, "Missing fraud label"
    assert 0 <= result["fraud_probability"] <= 1, "Probability must be valid"
