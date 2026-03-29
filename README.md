# impact_evaluation
Impact evaluation &amp; Econometrics
This library includes the following estimators:
- simple difference
- differences in differences
- matching
- regression discontinuity (RDD)
- control functions
- instrumental variables (IV)
- structural model

Moreover,  there is a recap of basic regression concepts.


### How each estimator "plugs in"
To keep the code clean, each class will interpret `**kwargs` differently. Here is how methods map to the code:

| Class Name | Key Parameter in `**kwargs` | Statistical Engine|
| ------------- | ------------- |  ------------- |
| SimpleDifference | None (Basic T-test/OLS) | `statsmodels.OLS`|
| DiffInDiff | `time_col, group_col` | Interaction Terms|
| Matching | `matching_type` (NN,  Kernel) | `sklearn.neighbors`|
| RDD | `running_variable,  cutoff` | Local Linear Regression|
| InstrumentalVariables | `instrument_col` | `linearmodels.IV2SLS`|
| ControlFunction | `first_stage_covariates` | Two-stage Residual Inclusion|
| StructuralModel | `model_equations,  params` | `scipy.optimize` / `PyMC`|
