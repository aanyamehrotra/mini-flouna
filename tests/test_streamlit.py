"""Exercise the presentation layer without changing the frozen runtime."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def test_default_request_displays_actual_recommendation():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    assert not app.exception
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert app.subheader[0].value == "Vegetable Biryani"
    assert app.metric[0].value == "₹270"
    assert app.metric[1].value == "4.7/5"
    assert app.metric[2].value == "24 min"


@pytest.mark.parametrize("example,heading", [
    ("Browse options", "Here are your options"),
    ("Ask a question", "Here's what we know"),
    ("Missing details", "A little more detail, please"),
    ("Nothing fits", "No matching option yet"),
    ("Unavailable information", "We can't verify that requirement"),
    ("An action request", "Ordering isn't supported in this demo"),
])
def test_examples_render_each_outcome(example, heading):
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    app.selectbox[0].select(example).run(timeout=20)
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert app.subheader[0].value == heading
    assert len(app.metric) == 0


def test_blank_input_has_a_friendly_message():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    app.selectbox[0].select("Write my own").run(timeout=20)
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert app.warning[0].value == "Please enter a food request first."
