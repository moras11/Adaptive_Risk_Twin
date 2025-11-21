def test_api_import():
    try:
        import src.api.paysim_api
    except Exception as e:
        assert False, f"API import failed: {e}"

    assert True
