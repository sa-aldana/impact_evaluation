import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_parallel_trends(did_model):
    """
    Visualizes the pre- and post-treatment means for DiD.
    Expects a DiffInDiff object with time_col and group_col in kwargs.
    """
    df = did_model.data
    time = did_model.kwargs.get('time_col')
    group = did_model.kwargs.get('group_col')
    outcome = did_model.outcome

    # Calculate means for the four quadrants
    means = df.groupby([time, group])[outcome].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=means, x=time, y=outcome, hue=group, marker='o')
    
    plt.title(f"Parallel Trends Check: {outcome}")
    plt.ylabel(f"Mean of {outcome}")
    plt.xlabel("Time (Pre vs Post)")
    plt.legend(title="Group", labels=['Control', 'Treated'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_effect_comparison(models_list):
    """
    Takes a list of fitted ImpactModel objects and plots their coefficients.
    Useful for comparing SimpleDifference vs. DiD vs. IV.
    """
    names = []
    effects = []
    errors = []

    for model in models_list:
        if model.results:
            names.append(model.__class__.__name__)
            # Extract effect (ATE or DiD_Effect)
            effect = model.results.get('ATE') or model.results.get('DiD_Effect')
            effects.append(effect)
            errors.append(model.results.get('Std_Error', 0))

    plt.figure(figsize=(8, 5))
    plt.errorbar(names, effects, yerr=errors, fmt='o', capsize=5, color='black')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Comparison of Estimated Treatment Effects")
    plt.ylabel("Estimated Impact")
    plt.show()
