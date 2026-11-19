"""TB fixes from the logic review: L-12 (detection rate capped at 100%), L-13 (WHO Table 5 blend for untreated HIV+ TB),
L-10 (no TB death trend: TB deaths are per patient), and the Nov 2026 redesign (TB in shares of the national TB
programme, deaths from the curve fitted to Clark et al. 2025)."""
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)
import health_model as hm  # noqa: E402

I, P = hm.load_inputs(), hm.load_params()
CI = I["ci"]
PC = hm.draw_params(P, 0)


def test_detection_rate_capped_at_100_and_flagged():
    row = CI.loc["KEN"].copy()
    row["sh_tbs_dtec_zs"] = 120.0
    ep = hm.epi(row, PC)
    assert ep["need_cov"]["tb_ds"][1] == 1.0
    assert any("capped at 100%" in f for f in ep["flags"])
    ok = hm.epi(CI.loc["KEN"], PC)
    assert ok["need_cov"]["tb_ds"][1] == pytest.approx(CI.loc["KEN", "sh_tbs_dtec_zs"] / 100)
    assert not any("TB case detection" in f for f in ok["flags"])


def test_untreated_hiv_tb_cfr_is_who_table5_blend():
    row = CI.loc["KEN"]
    a, h = CI.loc["KEN", "sh_hiv_artc_zs"] / 100, CI.loc["KEN", "tb_hiv_share"]
    got = hm.per_unit_levels(row, PC, a)["tb_ds"][0, 0]
    up = 0.78 * (1 - a) + 0.49 * a                         # WHO TB burden methods, Table 5
    tp = PC["tb_cfr_treated_pos"][0] * (1 - 0.3 * a)
    want = (1 - h) * (PC["tb_cfr_untreated_neg"][0] - PC["tb_cfr_treated_neg"][0]) + h * (up - tp)
    assert got == pytest.approx(want)
    assert PC["tb_cfr_untreated_pos"][0] == 0.78 and PC["tb_cfr_untreated_pos_art"][0] == 0.49
    # no one on ART: the off-ART value; everyone on ART for a year or more: the on-ART value
    for cov, cfr in ((0.0, 0.78), (1.0, 0.49)):
        lv = hm.per_unit_levels(row, PC, cov)["tb_ds"][0, 0]
        assert lv == pytest.approx((1 - h) * 0.40 + h * (cfr - PC["tb_cfr_treated_pos"][0] * (1 - 0.3 * cov)))


def test_no_tb_death_trend():
    assert "tb" not in hm.TREND_COLUMNS
    assert "trend_tb_deaths" not in CI.columns
    lv = hm.per_unit_levels(CI.loc["KEN"], PC, 0.9, trend=True)["tb_ds"]
    assert np.allclose(lv, lv[:, :1])                          # the same per-patient effect in every year


# --------------------------------------------------------------------------- #
# Nov 2026 redesign
# --------------------------------------------------------------------------- #
import pandas as pd  # noqa: E402
import scenarios as scn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FULL_TB = hm.uniform_scenario("TB", 1.0)


def _run(iso, inputs=I, ptab=P):
    return hm.run_country(iso, FULL_TB, hm.Fiscal(), inputs, ptab, n_draws=0)


def _tb_total(r):
    b = r["buckets"].set_index("bucket").loc["TB", "deaths_5y"]
    return b + r["committed"]["after_parts"]["tb"]


def test_tb_curve_is_the_clark_fit():
    fit = pd.read_csv(ROOT / "audit" / "logic_review" / "tb_clark_fit.csv")
    assert len(fit) == 22 and fit.within_ui.all()
    assert PC["tb_resp_lin"][0] == 0.371 and PC["tb_resp_quad"][0] == 0.736 and PC["tb_legacy_cfr"][0] == 0
    g = hm.tb_curve(PC, np.array([0.0, 0.1, 0.5, 1.0]))
    assert g[0] == 0 and np.all(np.diff(g) > 0)
    assert g[2] == pytest.approx(0.371 * 0.5 + 0.736 * 0.25)


@pytest.mark.parametrize("iso", ["VNM", "PHL", "IDN", "IND", "KEN"])
def test_full_cut_deaths_follow_the_programme_share(iso):
    """No replacement, sudden full cut: the share lost = donor TB money / programme funding, at most the external share,
    x (1 - continuity), every year; people not starting treatment = people starting now x that share; TB deaths
    2026-2035 = WHO TB deaths x the curve x the ramp-up."""
    r = _run(iso)
    row = CI.loc[iso]
    tb = float(I["lines"].loc[(I["lines"].iso3 == iso) & (I["lines"].bucket == "TB"), "usd"].sum())
    f = min(hm.tb_programme_share(row), tb / row["tb_prog_funding_usd"]) * (1 - PC["continuity"][0])
    L = r["lines"].set_index("line")
    need, cov = hm.epi(row, PC)["need_cov"]["tb_ds"]
    assert L.loc["tb_ds", "units_lost"] == pytest.approx(need * cov * f)
    want = hm.tb_deaths_base(row) * float(hm.tb_curve(PC, f)[0]) * hm.LAG["tb_ds"].sum()
    assert _tb_total(r) == pytest.approx(want)


def test_donor_money_beyond_the_programme_adds_no_deaths():
    """Viet Nam's TB aid can exceed its reported programme budget: once the donor TB money is more than the donor-funded
    part of the programme, more of it must not change what a full cut does (the share lost is capped at the external
    share), while less of it lowers the deaths (the share lost is money / the programme's funding)."""
    lines = I["lines"].copy()
    m = (lines.iso3 == "VNM") & (lines.bucket == "TB")
    k0 = 1.2 * CI.loc["VNM", "tb_prog_ext_share"] * CI.loc["VNM", "tb_prog_funding_usd"] / lines.loc[m, "usd"].sum()
    lines.loc[m, "usd"] *= k0                                       # just past the donor-funded part
    capped = {**I, "lines": lines}
    base = _tb_total(_run("VNM", capped))
    assert _tb_total(_run("VNM")) < base                            # the actual money is below the cap
    for k in (1.5, 3.0):
        more = capped["lines"].copy()
        more.loc[m, "usd"] *= k
        assert _tb_total(_run("VNM", {**I, "lines": more})) == pytest.approx(base)


def test_tb_share_uses_pooled_years_with_domestic_funding():
    """Single WHO years with no domestic funding reported (Viet Nam 2024, Pakistan 2024) are skipped."""
    t = pd.read_csv(ROOT / "audit" / "logic_review" / "tb_programme_share.csv").set_index("iso3")
    assert t.loc["VNM", "tb_prog_ext_year"] == "2022-2023" and t.loc["VNM", "tb_prog_ext_share"] < 0.95
    assert t.loc["PAK", "tb_prog_ext_year"] == "2022-2023" and t.loc["PAK", "tb_prog_ext_share"] < 0.95
    model = sorted(set(I["lines"].iso3) & set(CI.index))
    assert CI.loc[model, "tb_prog_ext_share"].between(0, 1).all()          # every model country has a value
    assert CI.loc["VNM", "tb_prog_ext_share"] == pytest.approx(t.loc["VNM", "tb_prog_ext_share"])


def test_legacy_switch_restores_the_per_patient_rule():
    """tb_legacy_cfr = 1: people losing treatment come from money / cost per course again (Philippines: far more TB
    deaths under a full cut than the programme-share rule gives)."""
    leg = P.copy()
    leg.loc["tb_legacy_cfr", ["central", "low", "high"]] = 1.0
    new, old = _run("PHL"), _run("PHL", ptab=leg)
    assert _tb_total(old) > 1.5 * _tb_total(new)


def test_tb_deaths_are_who_all_forms_not_double_counted():
    """WHO's all-forms TB deaths already include HIV-positive TB deaths (South Africa 2024: 54,000 = 25,000 + 29,000);
    tb_deaths_total and the TB curve's level use that figure, not all-forms + HIV-positive again."""
    zaf = CI.loc["ZAF"]
    assert zaf["tb_deaths_who"] == 54000 and hm.tb_deaths_base(zaf) == 54000
    assert zaf["tb_deaths_total"] == zaf["all_forms_of_tb_number_of_deaths_estimated"]
    lo, mid, hi = hm.tb_deaths_base(zaf, {"tb_deaths_unc": np.array([-1.0, 0.0, 1.0])})
    assert (lo, mid, hi) == (zaf["tb_deaths_who_lo"], 54000, zaf["tb_deaths_who_hi"])


def test_extra_deaths_per_person_not_starting_treatment_within_the_case_fatality_gap():
    """Physical bound, every country: a full cut can't kill more people a year than the people who stop starting
    treatment x the case-fatality gap between untreated and treated TB (the delay-not-denial curve gives far fewer)."""
    for iso in sorted(set(I["lines"].iso3) & set(CI.index)):
        r = _run(iso)
        L = r["lines"].set_index("line")
        lost = L.loc["tb_ds", "units_lost"]
        if lost <= 0:
            continue
        extra = (r["buckets"].set_index("bucket").loc["TB", "deaths_5y"] + r["committed"]["after_parts"]["tb"]) / 5
        dcfr = hm.per_unit_levels(CI.loc[iso], PC, hm.epi(CI.loc[iso], PC)["art_cov"])["tb_ds"][0, 0]
        assert extra / lost <= dcfr * (1 + 1e-9), iso
