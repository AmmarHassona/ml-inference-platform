from locust import HttpUser, task, between
import random

class PredictionUser(HttpUser):
    wait_time = between(0.5, 2)

    @task
    def predict(self):
        self.client.post("/predict",
            json={
                "features": [
                    round(random.uniform(4.3, 7.9), 1),
                    round(random.uniform(2.0, 4.4), 1),
                    round(random.uniform(1.0, 6.9), 1),
                    round(random.uniform(0.1, 2.5), 1),
                ]
            }
        )