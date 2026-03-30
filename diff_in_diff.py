import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

class DiffInDiff(ImpactModel):
    """
    Computes the Difference-in-Differences estimator.
    Requires 'time_col' (0=Pre, 1=Post) and 'group_col' (0=Control, 1=Treated) passed via kwargs.
    Formula: Y = B0 + B1*Time + B2*Group + B3*(Time*Group) + Covars
    """
    def estimate(self):
        time_col = self.kwargs.get('time_col')
        group_col = self.kwargs.get('group_col')
        
        if not time_col or not group_col:
            raise ValueError("DiffInDiff requires 'time_col' and 'group_col' in kwargs.")

        # Create the interaction term (The DiD Estimator)
        did_term = f"{time_col} * {group_col}"
        
        formula = f"{self.outcome} ~ {did_term}"
        if self.covariates:
            formula += " + " + " + ".join(self.covariates)
            
        model = smf.ols(formula, data=self.data).fit()
        self.model_fit = model
        
        # The impact is the coefficient of the interaction term
        # Statsmodels names interactions as 'time:group'
        interaction_key = f"{time_col}:{group_col}"
        
        self.results = {
            "DiD_Effect": model.params[interaction_key],
            "Std_Error": model.bse[interaction_key],
            "P_Value": model.pvalues[interaction_key],
            "R_Squared": model.rsquared
        }
        return self.results

