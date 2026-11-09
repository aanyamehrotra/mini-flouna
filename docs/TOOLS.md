# Libraries data and AI disclosure

## Tools and libraries

| Component | Use |
| --- | --- |
| Python | Application, scripts, JSON data handling, and extraction rules |
| scikit-learn | TF-IDF, Logistic Regression, classification metrics, and confusion matrix counts |
| joblib | Local model persistence |
| FastAPI | API routing and OpenAPI documentation |
| Pydantic | Request, response, catalogue, and label validation |
| Uvicorn | Local API server |
| pytest | Unit and API integration tests |
| HTTPX and FastAPI TestClient | In-process HTTP tests |
| Matplotlib | Exported test confusion matrix |
| NumPy and SciPy | Numerical dependencies used by scikit-learn and plotting |
| Git and GitHub | Version control and repository delivery |
| OpenAI Codex | Planning, implementation, synthetic data authorship and review, tests, documentation, and interpretation of results |

Direct dependency ranges are in `requirements.txt` and `requirements-dev.txt`. `requirements-evaluated.txt` captures the exact installed packages used for this run, including transitive dependencies. The recorded environment used Python 3.14.2 on macOS; that exact package snapshot requires a compatible Python/platform. For another supported Python version, use the bounded dependency files and report any resulting metric differences. Core numerical versions and data/model hashes also appear in evaluation metadata.

## Models APIs and data

The runtime model is a locally trained TF-IDF plus Logistic Regression pipeline. No pretrained transformer, runtime LLM, external AI generation API, production ONDC integration, payment provider, or live restaurant service is used. Interactive API documentation loads browser assets from a public CDN; request understanding and ranking use local resources.

The catalogue contains 30 fictional food options. The labelled request dataset contains 150 synthetic requests, including 60 preference annotations. Their creation methods, assumptions, and limitations are described in `data/README.md` and `data/REQUESTS.md`. No real customer records, private company data, credentials, or payment details are included.

## Authorship disclosure

OpenAI Codex assisted throughout the project, including code generation, dataset and gold-label authorship, AI review of labels, validation-driven rule refinements, tests, generated sample outputs, evaluation scripts, and documentation. There was no independent human annotation study. The code was run, tested, and evaluated locally; the recorded outputs are actual executions rather than fabricated scores.

The candidate should review and understand the implementation before submission and describe AI assistance accurately. This repository does not claim unaided implementation or independently human-authored labels.
