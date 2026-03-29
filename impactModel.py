from abc import ABC, abstractmethod
import pandas as pd

class ImpactModel(ABC):
    """
    Base Class for all Impact Evaluation Estimators.
    """
    def __init__(self, data: pd.DataFrame, outcome: str, treatment: str, **kwargs):
        self.data = data.copy()
        self.outcome = outcome
        self.treatment = treatment
        self.kwargs = kwargs  # Stores method-specific vars like 'instrument' or 'cutoff'
        self.model_fit = None
        self.results = {}

    @abstractmethod
    def estimate(self):
        """Logic for running the specific statistical estimator."""
        pass

    def get_summary(self):
        """Standardized output for all models."""
        if not self.results:
            return "Model has not been estimated."
        return pd.Series(self.results, name="Estimation Results")
