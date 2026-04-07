import pandas as pd
import numpy as np
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from ..base import ImpactModel

class EconMLDML(ImpactModel):
    """
    Modern Impact Evaluation using Double Machine Learning (DML).
    Uses Random Forests to handle non-linear relationships in covariates.
    """
    
    def estimate(self):
        # 1. DEFINE NUISANCE MODELS
        # model_y: Predicts outcome from covariates (Regression)
        # model_t: Predicts treatment from covariates (Classification/Propensity)
        model_y = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10)
        model_t = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10)
        
        # 2. INITIALIZE THE DML ESTIMATOR
        # discrete_treatment=True because our 'treatment' is binary (0 or 1)
        # cv=5 performs 5-fold cross-fitting to avoid overfitting bias
        est = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            cv=5
        )
        
        # 3. FIT THE MODEL
        # EconML expects X (covariates), T (treatment), and y (outcome)
        X = self.data[self.covariates].values
        T = self.data[self.treatment].values
        y = self.data[self.outcome].values
        
        est.fit(y, T, X=X)
        
        # 4. EXTRACT RESULTS
        # const_marginal_ate: This is our Average Treatment Effect
        ate = est.ate(X)
        # We can also get valid confidence intervals
        ci = est.ate_interval(X)
        
        self.results = {
            "ATE": ate,
            "Lower_CI": ci[0],
            "Upper_CI": ci[1],
            "Method": "Double Machine Learning (LinearDML)",
            "Engine": "EconML + Random Forest"
        }
        
        # Store the fitted object for advanced EconML diagnostics
        self.model_fit = est
        return self.results
