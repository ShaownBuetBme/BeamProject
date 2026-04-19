from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


def create_model(model_name: str, random_seed: int, alpha: float, n_estimators: int):
    model_name = model_name.lower().strip()

    if model_name == "ridge":
        return Ridge(alpha=alpha, random_state=random_seed)

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_seed,
            n_jobs=-1,
        )

    if model_name == "multioutput_ridge":
        # Kept explicit as a template for adding more wrapped estimators.
        return MultiOutputRegressor(Ridge(alpha=alpha, random_state=random_seed))

    raise ValueError(
        f"Unknown model_name '{model_name}'. Supported: ridge, random_forest, multioutput_ridge"
    )
