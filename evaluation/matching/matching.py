import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from ..base import ImpactModel

class Matching(ImpactModel):
    """
    Teaching implementation of PSM with support for:
    - N:1 Matching
    - Caliper (Radius) constraints
    """
    
    def estimate(self):
        # 1. SETUP PARAMETERS FROM KWARGS
        n_neighbors = self.kwargs.get('n_neighbors', 1)
        caliper = self.kwargs.get('caliper', None) # Max distance allowed
        
        # 2. PROPENSITY SCORE CALCULATION
        X = self.data[self.covariates]
        y = self.data[self.treatment]
        ps_model = LogisticRegression(penalty=None).fit(X, y)
        self.data['propensity_score'] = ps_model.predict_proba(X)[:, 1]
        
        # 3. SEPARATE GROUPS
        treated = self.data[self.data[self.treatment] == 1].copy()
        control = self.data[self.data[self.treatment] == 0].copy()
        
        # 4. NEAREST NEIGHBOR SEARCH
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        nn.fit(control[['propensity_score']])
        
        # distances: array of distances to neighbors
        # indices: indices of neighbors in the 'control' dataframe
        distances, indices = nn.kneighbors(treated[['propensity_score']])
        
        # 5. APPLY CALIPER (IF PROVIDED)
        # We mask out any matches where the distance > caliper
        if caliper is not None:
            mask = distances[:, 0] <= caliper
            # Filter treated and their corresponding neighbors
            treated_filtered = treated[mask]
            indices_filtered = indices[mask]
            dropped_units = len(treated) - len(treated_filtered)
        else:
            treated_filtered = treated
            indices_filtered = indices
            dropped_units = 0

        # 6. CALCULATE ATT
        # For N:1, we average the outcomes of the N neighbors for each treated unit
        control_outcomes = control[self.outcome].values
        
        # Get mean outcome of matched controls for each treated unit
        matched_values = np.array([np.mean(control_outcomes[idx]) for idx in indices_filtered])
        treated_values = treated_filtered[self.outcome].values
        
        att = np.mean(treated_values - matched_values)
        
        # 7. STORE RESULTS
        self.results = {
            "ATT": att,
            "N_Neighbors": n_neighbors,
            "Caliper_Used": caliper,
            "Units_Dropped_By_Caliper": dropped_units,
            "Final_Sample_Size": len(treated_filtered)
        }
        return self.results
