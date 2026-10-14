"""Scaling + feature selection (built as one scikit-learn Pipeline)."""
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import TOP_K_FEATURES, RANDOM_STATE


def build_pipeline():
    """StandardScaler -> SelectKBest(ANOVA F-test) -> MLP (Adam, L2)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=TOP_K_FEATURES)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,              # L2 regularization
            learning_rate_init=0.001,
            max_iter=300,
            random_state=RANDOM_STATE)),
    ])


def selected_feature_names(pipeline, feature_names):
    mask = pipeline.named_steps["select"].get_support()
    return [f for f, keep in zip(feature_names, mask) if keep]
