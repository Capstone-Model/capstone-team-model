"""Tests for the allocation model (allocation_model.py) and the Reallocation page. Run:  python -m pytest tests

The workbook's own results PDF was reproduced exactly by running allocation_model.py on the engine as committed when
the PDF was made (deaths 898,897 / 199,893 / 0 / 506,894; lambda 36.6%; 19 countries at zero aid). These tests guard
the parts that do not depend on the engine version, and check the saved results are in step with the current engine.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)
import allocation_model as am  # noqa: E402
import health_model as hm  # noqa: E402

W = am.read_workbook()
R = am.load()
C = R["countries"]


def test_capacity_matches_workbook_formulas():
    """Values the workbook's formulas give (checked with an independent formula evaluator)."""
    cap = am.capacity(W)
    assert cap.at["NGA", "capacity"] == pytest.approx(299.754, abs=1e-3)
    assert cap.at["AFG", "capacity"] == pytest.approx(30.0439, abs=1e-3)
    assert cap.at["MOZ", "capacity"] == pytest.approx(21.3448, abs=1e-3)
    assert cap.at["KSV", "capacity"] == 0            # no health share and no fiscal space score
    assert cap["capacity"].sum() == pytest.approx(64704.58, abs=0.01)


def test_solver_keeps_the_pool_and_caps_cuts_at_aid():
    total_cut = C["aid_2023"].sum() - C["aid_2026_proj"].sum()
    assert total_cut == pytest.approx(4716.02, abs=0.01)
    for rule in am.RULES:
        cut, aid = C[f"cut_{rule}"], C[f"aid_2026_{rule}"]
        assert cut.sum() == pytest.approx(total_cut, abs=1e-6)
        assert aid.sum() == pytest.approx(C["aid_2026_proj"].sum(), abs=1e-6)
        assert (cut <= C["aid_2023"] + 1e-9).all() and (cut >= 0).all()
        lam = R["rules"].at[rule, "lambda"]
        inside = cut < C["aid_2023"] - 1e-9            # countries not cut to zero lose lambda x effective capacity
        assert np.allclose(cut[inside], lam * C.loc[inside, f"eff_capacity_{rule}"])


def test_rules_and_zero_aid_counts():
    r = R["rules"]
    assert r.at["cpl_need", "lambda"] == pytest.approx(0.366, abs=5e-4)
    assert r.at["cap", "lambda"] == pytest.approx(0.0900, abs=5e-4)
    assert r["zero_aid"].to_dict() == {"cpl_need": 19, "cpl": 16, "need": 13, "cap": 12}


def test_need_tiers_reproduce_the_funding_gap_chart():
    assert C["need_tier"].value_counts().to_dict() == {"Low": 30, "High": 27, "Medium": 25}
    pc = C["aid_2023"] - C["aid_2026_proj"]
    assert pc[C["need_tier"] == "High"].sum() == pytest.approx(2726.7, abs=0.1)


def test_new_money_closes_every_gap_and_scenarios_are_ordered():
    d = am.deaths_by_disease(R).loc["Total"]
    assert d["green"] == pytest.approx(0, abs=1e-6)
    assert d["red"] > d["blue"] > d["yellow"] > 0
    assert (C["cut_cpl_need"] <= C["capacity"] + 1e-9).all()       # so the reallocated + new money gap is zero


def test_coverage_with_no_losses_is_baseline():
    s = R["services"]
    g = s[(s.rule == am.PLAN) & (s.scenario == "green") & (s.need > 0) & s.cov0.notna()]
    base = (g.need * g.cov0).groupby(g.line).sum() / g.need.groupby(g.line).sum()
    assert np.allclose(am.coverage(R)["green"].reindex(base.index), base)


def test_saved_results_match_the_current_engine():
    """Fails when the engine has changed since model_data/allocation was written: run python allocation_model.py."""
    I, P = hm.load_inputs(), hm.load_params()
    isos = ["AGO", "ZMB", "NGA", "IDN"]
    now = am.cost_per_life(isos, I, P)
    saved = R["cost_per_life"].reindex(isos)
    for b in am.BUCKETS:
        assert np.allclose(now[f"l_{b}"], saved[f"l_{b}"], equal_nan=True), b
    iso = "KEN"
    rate = C.at[iso, f"cut_{am.PLAN}"] / C.at[iso, "aid_2023"]
    r = am.run_scenario(iso, rate, 0.0, I, P)
    blue = R["disease"].query("rule == @am.PLAN and scenario == 'blue' and iso3 == @iso")["deaths_5y"].sum()
    assert r["totals"]["deaths_5y"] == pytest.approx(blue, rel=1e-9)


@pytest.mark.parametrize("rule", list(am.RULES))
def test_page_renders_for_every_rule(rule):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "views" / "5_Allocation.py"),
                           default_timeout=120)
    at.session_state["al_rule"] = rule
    at.run()
    assert not at.exception
    for iso in ["KSV", "VNM", "NER"]:                  # zero capacity, zero aid, most efficient
        at.selectbox(key="al_country").set_value(iso).run()
        assert not at.exception
