import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

class SimpleDifference(ImpactModel):
    """
    This class computes the simple difference estimator (in means) between Treatment and Control.
    It equates to a simple OLS: Y = alpha + beta*D + epsilon, where D is program participation
    """
    def estimate(self):
        # Prepare the formula: outcome ~ treatment + covariates
        formula = f"{self.outcome} ~ {self.treatment}"
        if self.covariates:
            formula += " + " + " + ".join(self.covariates)
        
        model = smf.ols(formula, data=self.data).fit()
        self.model_fit = model
        
        # Extract the coefficient for the treatment variable
        self.results = {
            "ATE": model.params[self.treatment],
            "Std_Error": model.bse[self.treatment],
            "P_Value": model.pvalues[self.treatment],
            "R_Squared": model.rsquared
        }
        return self.results

  
