"""The IHME / OECD aid data sources (announcement_model/workbook_inputs.py -> model_data/sources/<source>/): the money
the model runs on must reconcile with each projection workbook, cell by cell and country by country.
Run:  python -m pytest tests/test_sources.py"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "announcement_model"))
logging.disable(logging.WARNING)
import health_model as hm  # noqa: E402
import scenarios as scn  # noqa: E402
import workbook_inputs as wi  # noqa: E402
from prepare_model import DASHBOARD_ISO3  # noqa: E402

YRS = list(range(2026, 2031))
pytestmark = pytest.mark.skipif(not all(p.exists() for p in wi.WORKBOOKS.values()),
                                reason="projection workbooks not present")


@pytest.fixture(scope="module", params=list(hm.SOURCES))
def src(request):
    s = request.param
    rows = wi.load_rows(s)
    return {"source": s, "rows": rows, "I": hm.load_inputs(source=s),
            "br": pd.read_excel(wi.WORKBOOKS[s], "By_Recipient").dropna(subset=["ISO3"])}


def test_headline_total_matches_workbook(src):
    """All-health aid per country (the money headline) = the workbook's By_Recipient baseline and projections."""
    tot = hm.announcement_totals(source=src["source"])
    br = src["br"].assign(ISO3=lambda d: d.ISO3.replace(wi.ISO_FIX)).set_index("ISO3")
    for iso in [c for c in DASHBOARD_ISO3 if c in tot]:
        assert tot[iso]["base"] == pytest.approx(br.at[iso, "Baseline DAH (k$)"] * 1e3, rel=1e-6)
        proj = br.loc[iso, [f"Proj {y}" for y in YRS]].to_numpy(float) * 1e3
        np.testing.assert_allclose(tot[iso]["base"] - tot[iso]["lost"], proj, rtol=1e-5, atol=5)


def test_disease_money_matches_workbook(src):
    """HIV, TB and malaria money in the model's cells = the workbook rows' baseline x their health-area share (OECD),
    or the rows' own IHME program areas (IHME), for every dashboard country."""
    cells, rows = src["I"]["lines"], src["rows"]
    rows = rows[rows.iso3.isin(DASHBOARD_ISO3)]
    got = cells.groupby(["iso3", "bucket"]).usd.sum()
    if src["source"] == "oecd":
        for b, col in wi.AREA.items():
            # a few OECD rows carry a negative share (net adjustments, e.g. WHO own funds to South Sudan, -US$48K of
            # malaria): there is no negative money to cut, so those are left out
            want = (rows.Baseline * rows[col]).clip(lower=0).groupby(rows.iso3).sum() * 1e3
            g = got.xs(b, level="bucket").reindex(want.index).fillna(0)
            np.testing.assert_allclose(g, want, rtol=1e-6, atol=1)
    else:
        raw = wi.raw_ihme_2023()
        m = rows.merge(raw, on=["iso3", "source", "channel"], how="left").fillna(0)
        for b in hm.BUCKETS:
            lines = [l for l in wi.LN if wi.BKT[l] == b]
            want = m[lines].sum(axis=1).groupby(m.iso3).sum()
            g = got.xs(b, level="bucket").reindex(want.index).fillna(0)
            np.testing.assert_allclose(g, want, rtol=1e-6, atol=1)


def test_cell_cuts_reproduce_workbook_loss(src):
    """Each cell's yearly loss (usd x cut) adds up to the workbook's projected loss on that money: per country, the
    four diseases' loss = sum over rows of (Baseline - Proj) x the row's share on those service lines."""
    cells, rows = src["I"]["lines"], src["rows"]
    rows = rows[rows.iso3.isin(DASHBOARD_ISO3)]
    split = wi.split_ihme(rows, wi.raw_ihme_2023()) if src["source"] == "ihme" else wi.split_oecd(rows, wi.raw_ihme_2023())
    share = split[wi.LN].sum(axis=1) / (split.Baseline * 1e3)
    for y in YRS:
        want = ((split.Baseline - split[f"Proj {y}"]) * 1e3 * share).groupby(split.iso3).sum()
        got = (cells.usd * cells[f"cut_{y}"]).groupby(cells.iso3).sum().reindex(want.index).fillna(0)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=50)


def test_announced_preset_uses_cell_cuts(src):
    """The Announced donor cuts preset takes each cell's own yearly cut, so the model's gross loss for a country is
    the cells' usd x cut (clipped at zero by service line, since gains aren't modelled)."""
    I, P = src["I"], hm.load_params()
    sc = scn.build_scenario(scn.ANNOUNCED, {}, I["ci"])
    r = hm.run_country("KEN", sc, hm.Fiscal(*scn.DEFAULT_FISCAL), I, P, n_draws=0)
    c = I["lines"][I["lines"].iso3 == "KEN"]
    want = np.mean([(c.usd * c[f"cut_{y}"]).groupby(c.line).sum().clip(lower=0).sum() for y in YRS])
    assert r["totals"]["gross"] == pytest.approx(want, rel=1e-6)
    assert r["totals"]["base"] == pytest.approx(c.usd.sum(), rel=1e-9)


def test_sources_differ_and_share_country_data():
    """Switching source changes the money but not the country epidemiology."""
    a, b = hm.load_inputs(source="ihme"), hm.load_inputs(source="oecd")
    assert a["ci"].equals(b["ci"])
    ka, kb = (x["lines"].query("iso3 == 'KEN'").usd.sum() for x in (a, b))
    assert ka > 0 and kb > 0 and abs(ka / kb - 1) > 0.01
