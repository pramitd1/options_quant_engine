from sklearn.ensemble import RandomForestClassifier


class MovePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        prob = self.model.predict_proba(X)
        return prob[:, 1]

    def predict_probability(self, X):
        prob = self.model.predict_proba(X)
        return prob[:, 1]