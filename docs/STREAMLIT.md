# Streamlit demo and deployment

The Streamlit interface calls the existing request-processing pipeline directly. It adds no new recommendation rules, external data API, or transaction behaviour. The frozen test results remain results of the evaluated runtime, not a claim about a new model.

## Local preview

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open http://localhost:8501. Choose an example or write a request, then click **Get result**. The page shows the recommendation or appropriate alternative outcome, available options, extracted preferences, full JSON, and a JSON download. No FastAPI server is required for this interface.

## Model preparation

`streamlit_app.py` verifies the dataset and split hashes against `reports/validation.json`. It then fits the already-selected C value on the 90 training requests and uses the selected clarification threshold. Streamlit caches this tiny model once per process. It does not tune parameters, refit using validation/test requests, alter reports, or require a binary model to be uploaded. This avoids cross-version pickle loading during deployment.

For the command-line API workflow, the existing `python -m scripts.train` command remains available. The Streamlit wrapper was tested using Streamlit 1.67.0 and Python 3.14.2. The earlier `requirements-evaluated.txt` remains the numerical benchmark environment snapshot and predates the additional Streamlit dependencies; normal installation uses the updated `requirements.txt`.

## Community Cloud settings

Sign in at https://share.streamlit.io and create an app from the existing GitHub repository:

| Setting | Value |
| --- | --- |
| Repository | `aanyamehrotra/mini-flouna` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| Python version | `3.14`, matching local testing, if available in Advanced settings |
| Secrets | None |
| Dependencies | Root `requirements.txt` |

Choose an available app subdomain in the deployment form. Only the actual URL issued by Streamlit after successful deployment is a shareable public link. A localhost preview is not a public deployment. Account sign-in, terms acceptance, and any new GitHub access authorization must be completed or approved by the account owner.

If the selected Python version is unavailable, use a supported version and verify the sample requests after deployment. Bounded dependencies allow the platform to resolve compatible packages; exact benchmark reproduction depends on the recorded numerical versions.

Official deployment reference: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
