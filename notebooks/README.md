# Notebook examples

Start with `01_tweedie_revenue_pipeline.ipynb` for the current revenue workflow.
It covers real-data loading and column mapping, original euro versus cent units,
discount units, optional calendar features, pretreatment X/W, group splitting,
Tweedie configuration, fitting, diagnostics, euro and percentage interpretation,
cohort reports and standalone reuse. It requires your actual dataframe.

`02_api_examples.ipynb` collects commented snippets from the earlier Markdown
guides. They are disabled by default because they refer to user-supplied arrays
and dataframes. Uncomment and adapt an example only after defining its inputs.

`run_pipeline.ipynb` preserves the original R-learner/log-outcome example.
`run_recommendations.ipynb` preserves the general categorical DR example.
Their root-level files are now Markdown-only pointers, keeping old links useful.

Training the Tweedie nuisance uses nonnegative revenue in original currency.
The log link is internal. Final causal effects and DR scores can be signed;
do not give them a Tweedie objective. Revenue means and policy values are not
full predictive distributions, and Tweedie does not guarantee outlier resistance.

The examples locate the checkout from either the repository root or this
directory. Historical examples switch the working directory to the root;
the current Tweedie notebook uses explicit root-based paths.
