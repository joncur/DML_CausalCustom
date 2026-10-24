# Run the synthetic model test in Codespaces

1. Open https://github.com/joncur/DML_CausalCustom and select the `main` branch.
2. Select **Code → Codespaces → Create codespace on main**. For this small test, start with the smallest offered machine. GitHub account quotas/billing apply.
3. Wait for the container setup to finish. The committed `.devcontainer.json` selects Python 3.11, installs `requirements-notebooks.txt`, and requests the Python and Jupyter editor extensions. It does not start training automatically.
4. In the left file explorer open **notebooks/00_synthetic_model_test.ipynb**.
5. At the top right choose **Select Kernel → Python Environments → Python 3.11** (the configured `/usr/local/bin/python`, labels may vary). Select **Run All**.
6. Read the effect table, plot, held-out policy table, diagnostic values and PASS messages. The final cell runs the repository unit suite; the expected current result is **42 passed**, followed by **ALL ENABLED CHECKS PASSED**.

The notebook generates 8,000 synthetic views from 4,000 users. One segment has a true carousel effect of +€4/view, another −€2/view. It tests reference direction, `const_marginal_effect`, grouped splits, effect recovery, independent policy evaluation, euros/percent consistency, and save/reload equivalence. Changing data settings is an experiment and can invalidate its fixed smoke-test tolerances. Passing does not validate observational identification on your real data.

You do **not** close the repository or manually clone it after creating the Codespace: Codespaces checks it out for you. GitHub's notebook page is a read-only preview; code executes in the Codespaces editor. No separate Jupyter server or port forwarding is needed for VS Code notebook execution.

## Existing Codespace or installation trouble

An existing Codespace may have an older checkout. Save or commit your work, then use Source Control to pull the latest `main` changes. Run **Codespaces: Rebuild Container** from the command palette to apply the new configuration. If you prefer keeping your current container, run the notebook's setup cell, which installs missing dependencies using the active kernel. If setup was incomplete or versions conflict, use the terminal command below from the repository root, then restart the notebook kernel and Run All:

```bash
python -m pip install -r requirements-notebooks.txt
```

To run just the unit suite in the terminal:

```bash
python -m pytest -q
```

All runnable Python demonstration snippets remain in the notebooks directory.

## Save and stop

Use **Ctrl+S** or **File → Save**. To persist edits in GitHub, commit and sync them in Source Control; saving alone does not push. Generated model/check files go to ignored `outputs/synthetic_smoke_test/`; they are not committed automatically. Download any generated artifacts you need before deleting a Codespace.

When finished, open the command palette (**Ctrl+Shift+P**, or **View → Command Palette**) and choose **Codespaces: Stop Current Codespace**. Alternatively, open https://github.com/codespaces, use the Codespace's three-dot menu, and choose **Stop codespace**. Closing the browser tab alone may leave compute running until timeout. Stopped Codespaces preserve work for resumption, but storage limits/charges and retention settings still apply. Reopen the same Codespace from the Codespaces page next time.

## Documentation

- [Create a Codespace for a repository](https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository)
- [Set up Python](https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/setting-up-your-python-project-for-codespaces)
- [Stop and restart](https://docs.github.com/en/codespaces/developing-in-a-codespace/stopping-and-starting-a-codespace)
