from locust import HttpUser, task, between
import random

class PredictionUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(2)
    def predict(self):
        self.client.post("/predict",
            json={
                "features": [
                    round(random.uniform(17, 90), 1),    # age
                    round(random.uniform(12285, 1484705), 1),  # fnlwgt
                    round(random.uniform(1, 16), 1),     # education-num
                    round(random.uniform(0, 99999), 1),  # capital-gain
                    round(random.uniform(0, 4356), 1),   # capital-loss
                    round(random.uniform(1, 99), 1),     # hours-per-week
                ]
            }
        )
    
    @task(1)
    def predict_text(self):
        self.client.post("/predict/text", json = {"text": "If you're reading this, have a nice day :)"})