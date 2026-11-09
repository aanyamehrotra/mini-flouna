# Live demo guide

Open the [live Streamlit demo](https://mini-flouna-kvokv5gr9m899aqnzdfhvr.streamlit.app/). Choose an example and click **Get result**. The application uses fictional food options and needs no API keys. It is the live demo deliverable; no prerecorded video is included.

## Run a local demo

After the root README setup, run the browser interface:

```bash
python -m streamlit run streamlit_app.py
```

Alternatively, run the API:

```bash
python -m scripts.train
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs. Expand `POST /recommend`, click **Try it out**, select an example, and click **Execute**. Requests have just one field: `text`.

## Suggested three-minute walkthrough

1. **Explain the scope.** “This is a small food request-understanding component. It combines a text classifier with bounded extraction rules and an explainable ranking policy. All catalogue data is fictional.”
2. **Run the Decision example.** The highly rated vegetarian biryani request under ₹300 and within 30 minutes recommends Garden Pot's ₹270 item with restaurant rating 4.7 and estimated delivery in 24 minutes. Expand extracted preferences and ranking contributions. Point out that the ₹200 option is eligible but not the best fit under this policy.
3. **Run Discovery.** “Show vegetarian pizza” returns options without a selected recommendation. Its budget remains `null`; a missing optional preference does not block the request. A warning explains that no locality was specified.
4. **Show uncertainty and unsupported data.** “Help me” asks for clarification. Organic pizza reports that organic status cannot be verified. Vegetarian biryani under ₹50 returns no match rather than increasing the budget.
5. **Demonstrate the Action boundary.** “How do I place an order?” returns Information. “Please order a pizza” identifies Action but returns `unsupported_action`; nothing is executed. “What should I order?” also never authorizes ordering; the current pipeline asks for more food context.
6. **Show evaluation and limitations.** Open `reports/test_confusion_matrix.png` and `reports/README.md`. Explain the raw classifier's 63.3% test accuracy versus the full pipeline's 93.3%, the 0/24 observed false Actions, and the small synthetic dataset. Show at least one documented failure rather than claiming perfect understanding.

Review the source and the README's AI-use disclosure before presenting the implementation.

## Sample outputs

`examples/sample_io.json` contains eight complete outputs captured from the final frozen pipeline, including all seven response statuses. These are demo samples, separate from the labelled test benchmark. The API documentation also provides seven runnable request examples.

## Review without a browser

```bash
python -m app "Show vegetarian pizza"
python -m app "Recommend vegetarian pizza under ₹300"
python -m app "Please order a pizza"
python -m scripts.evaluate
python -m pytest -q
```

The recorded test errors remain reproducible. No runtime rule changes were made to improve the final test score.
