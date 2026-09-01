import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture
def fresh_app():
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


def test_app_loads_without_error(fresh_app):
    assert not fresh_app.exception


def test_default_candidate_pool_is_10(fresh_app):
    metrics = [m.value for m in fresh_app.metric]
    assert metrics[0] == "10"


def test_default_mean_experience(fresh_app):
    metrics = [m.value for m in fresh_app.metric]
    assert "9.4" in metrics[2]
def test_region_filter_runs_without_error():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.selectbox[2].select("Europe").run()
    assert not at.exception


def test_zero_result_edge_case():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.selectbox[2].select("Europe").run()
    at.multiselect[3].select("Energy").run()
    at.multiselect[3].select("Technology").run()
    at.slider[0].set_range(6.5, 11.5).run()
    assert not at.exception
    metrics = [m.value for m in at.metric]
    assert metrics[1] == "0"


def test_preset_mandate_sets_top_match():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.selectbox[1].select("US Healthcare Fundamental Analyst (5-10 yrs)").run()
    assert not at.exception
    metrics = [m.value for m in at.metric]
    assert metrics[3] != "-"


def test_all_tabs_load_without_error():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    for i in range(len(at.tabs)):
        at.tabs[i].run()
        assert not at.exception