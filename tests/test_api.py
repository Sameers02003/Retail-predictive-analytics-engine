REVIEW_ID = "T00062458"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["database_connected"] is True


def test_forecast_returns_prediction_with_version_and_timestamp(client):
    response = client.get("/forecast", params={"store_id": "S009", "date": "2026-01-03"})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_sales"] > 0
    assert body["model_version"] == "forecast_lgbm_v1"
    assert body["predicted_at"]
    assert len(body["features_used"]) == 11


def test_forecast_is_saved_to_sql(client, conn):
    def saved_rows():
        try:
            return conn.execute("SELECT COUNT(*) FROM api_forecast_log WHERE store_id = 'S005'").fetchone()[0]
        except Exception:
            return 0

    before = saved_rows()
    client.get("/forecast", params={"store_id": "S005", "date": "2026-01-02"})
    assert saved_rows() == before + 1


def test_unknown_store_gives_404(client):
    assert client.get("/forecast", params={"store_id": "S099", "date": "2026-01-03"}).status_code == 404


def test_date_beyond_horizon_gives_422(client):
    response = client.get("/forecast", params={"store_id": "S009", "date": "2026-02-01"})
    assert response.status_code == 422
    assert "at most 7 days" in response.json()["detail"]


def test_badly_formatted_store_gives_422(client):
    assert client.get("/forecast", params={"store_id": "s009", "date": "2026-01-03"}).status_code == 422


def test_evaluation_lists_both_models(client):
    models = {row["model"] for row in client.get("/forecast/evaluation").json()}
    assert models == {"seasonal_naive", "lightgbm"}


def test_anomalies_always_include_disclaimer(client):
    body = client.get("/anomalies", params={"priority": "High", "limit": 5}).json()
    assert body["count"] == 5
    assert "NOT proof of fraud" in body["disclaimer"]
    assert all(item["priority"] == "High" and item["reason"] for item in body["anomalies"])


def test_review_is_saved(client):
    payload = {"review_status": "Dismissed", "reviewed_by": "Test", "review_note": "test run"}
    response = client.patch(f"/anomalies/{REVIEW_ID}/review", json=payload)
    assert response.status_code == 200
    assert response.json()["review_status"] == "Dismissed"
    dismissed = client.get("/anomalies", params={"review_status": "Dismissed", "limit": 500}).json()
    assert REVIEW_ID in [item["transaction_id"] for item in dismissed["anomalies"]]


def test_invalid_review_status_gives_422(client):
    payload = {"review_status": "Maybe", "reviewed_by": "Test"}
    assert client.patch(f"/anomalies/{REVIEW_ID}/review", json=payload).status_code == 422


def test_review_of_unknown_transaction_gives_404(client):
    payload = {"review_status": "Dismissed", "reviewed_by": "Test"}
    assert client.patch("/anomalies/NOPE/review", json=payload).status_code == 404