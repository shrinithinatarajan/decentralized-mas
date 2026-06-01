import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestBaseline:
    def __init__(self, random_state: int | None = None) -> None:
        self._model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Model is not fitted — call fit() first.")
        return ["SENSITIVE" if p == 1 else "RESISTANT" for p in self._model.predict(X)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted — call fit() first.")
        return self._model.predict_proba(X)[:, 1]
