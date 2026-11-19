"""
TB pathway redesign (Nov 2026): the inputs for counting TB deaths the way the LSHTM TB model does (Clark et al., Lancet
Glob Health 2025; doi 10.1016/S2214-109X(25)00232-3). Read-only with respect to the model.

    python audit/tb_programme.py

Clark et al. assume that a funding cut lowers the rate at which people start TB treatment in proportion to the share of
the national TB programme's budget that is lost, and run a calibrated transmission model to 2035. They publish results
by income group and WHO region, not by country. This script does two things:

1. External share of each model country's national TB programme (NTP) funding, from WHO's TB financing data
   (datasets/WHO_TB/TB_expenditure_utilisation_2026-10-14.csv, funding received):
       (rcvd_tot_gf + rcvd_tot_usaid + rcvd_tot_grnt) / rcvd_tot_sources,
   pooled over the latest two years 2021-2024 in which the country reports domestic funding above zero (single years
   jump: Viet Nam reports no domestic funding for 2024, Pakistan reports zero). If no year reports domestic funding: the
   same from the budget file's expected funding (TB_budget_2026-10-14.csv, cf_*, 2022-2025); else the latest received
   year 2021-2024 if it reports domestic funding as zero (a blank is not reported, not zero); else the median of model countries in the WHO region (at least 3), else of all model countries.
   tb_prog_funding_usd = the programme's funding a year from all sources over the same years (US$; blank for a
   regional median): the model divides donor TB money lost by it, as the study does, and caps the result at the
   external share (so donor money recorded beyond the programme, as in Viet Nam, can't remove more than the
   donor-funded part). Clark et al.'s own value (the 2023 USAID + Global Fund share of expected funding, from their repository's
   funding_scenario_values.csv, scen_usaid_gf) is kept alongside as a comparison.

2. The curve linking the share of the NTP budget lost (f) to the rise in TB deaths, fitted to all 22 death results
   Clark et al. publish (Table 1: USAID termination; all expected Global Fund cuts; full Global Fund termination; the US
   expected and full cuts. Table 2: by income group and region for the first two):
       rise in TB deaths (share of baseline deaths, 2025-2035) = a f + b f^2
   Each published group value is predicted from Clark et al.'s country cuts (their repository, scenario columns) as the
   death-weighted average over the group's countries (WHO 2023 TB deaths, TB_burden_countries); results reported
   relative to the USAID scenario are (g(x) - g(u)) / (1 + g(u)) per country. Weighted least squares with each value's
   95% interval as its spread; iterated because the denominator depends on (a, b).
   The LSHTM repository (github.com/lshtm-tbmg/tbmod-pub-funding-impacts, commit b5f86fa, Jan 2026) has no licence
   file, so its two tables are read from datasets/LSHTM_TB/ (not committed) and only the fit is written here.

3. WHO's estimated TB deaths (all forms, HIV-negative + HIV-positive; latest year, with its 95% interval) from
   TB_burden_countries, the level the deaths curve scales (tb_deaths_who, _lo, _hi, _year).

Data screens: a single year whose total from all sources is over 4 times, or under a quarter of, the country's
2019-2025 median is dropped (Timor-Leste 2024, Botswana 2022); where received funding is below the programme's
expected funding over the same years and the domestic amount is copied forward unchanged, received reporting is
incomplete and the expected funding is used (Iraq, South Africa).

Writes audit/logic_review/tb_programme_share.csv and audit/logic_review/tb_clark_fit.csv; prints the fit.
batch34_inputs.py --tb-programme copies the share into model_data/country_inputs.csv; the fit's a and b are the central
values of tb_resp_lin and tb_resp_quad in model_params.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WHO_TB = ROOT / "datasets" / "WHO_TB"
LSHTM = ROOT / "datasets" / "LSHTM_TB"
OUT_SHARE = ROOT / "audit" / "logic_review" / "tb_programme_share.csv"
OUT_FIT = ROOT / "audit" / "logic_review" / "tb_clark_fit.csv"
EXT = ["gf", "usaid", "grnt"]

# Clark et al. 2025, Tables 1-2: (scenario column, reference column or None, group, % rise in deaths, 95% UI low, high)
PUBLISHED = [("scen_usaid", None, g, v, lo, hi) for g, v, lo, hi in [
    ("All", 5.0, 4.1, 5.9), ("LIC", 18.1, 13.8, 23.3), ("LMIC", 3.7, 2.8, 4.6), ("UMIC", 2.0, 0.6, 3.2),
    ("AFR", 12.1, 9.6, 14.7), ("EMR", 1.4, 0.9, 2.1), ("EUR", 1.7, -0.8, 4.4), ("SEAR", 2.7, 1.8, 3.5),
    ("WPR", 2.9, -0.1, 6.1)]]
PUBLISHED += [("scen_usaid_gf_expcut_noNGO", "scen_usaid", g, v, lo, hi) for g, v, lo, hi in [
    ("All", 12.2, 9.2, 14.6), ("LIC", 22.0, 17.7, 27.4), ("LMIC", 10.3, 8.0, 13.0), ("UMIC", 12.1, 2.7, 19.9),
    ("AFR", 15.6, 12.1, 19.2), ("AMR", 0.8, -0.1, 1.6), ("EMR", 32.1, 22.0, 42.4), ("EUR", 4.5, -1.3, 10.8),
    ("SEAR", 9.5, 4.5, 13.3), ("WPR", 7.2, 2.3, 11.0)]]
PUBLISHED += [("scen_usaid_gf", "scen_usaid", "All", 38.1, 27.0, 47.8),
              ("scen_usaid_USA_expcut", "scen_usaid", "All", 8.8, 6.6, 10.4),
              ("scen_usaid_USA", "scen_usaid", "All", 9.6, 7.2, 11.4)]


def _screen(df: pd.DataFrame, pre: str) -> pd.DataFrame:
    """Drops implausible single years: a total from all sources more than 4 times, or under a quarter of, the
    country's median over 2019-2025 (Timor-Leste reports US$309M of Global Fund money for 2024 against US$2-10M in
    other years; Botswana US$0.9M for 2022 against a median of US$9M)."""
    d = df[df[f"{pre}_tot_sources"] > 0].copy()
    med = d[d.year >= 2019].groupby("iso3")[f"{pre}_tot_sources"].median()
    m = d.iso3.map(med)
    bad = (d[f"{pre}_tot_sources"] > 4 * m) | (d[f"{pre}_tot_sources"] < m / 4)
    return d[~bad]


def _pooled(df: pd.DataFrame, pre: str, years: range) -> pd.DataFrame:
    """External share pooled over the latest two years in `years` with domestic funding reported above zero."""
    df = _screen(df, pre)
    d = df[df.year.isin(years) & (df[f"{pre}_tot_sources"] > 0) & (df[f"{pre}_tot_domestic"] > 0)].copy()
    d["ext"] = d[[f"{pre}_tot_{k}" for k in EXT]].sum(axis=1, min_count=1).fillna(0.0)
    d = d.sort_values("year").groupby("iso3").tail(2)
    g = d.groupby("iso3").agg(ext=("ext", "sum"), tot=(f"{pre}_tot_sources", "sum"),
                              y0=("year", "min"), y1=("year", "max"), ny=("year", "size"))
    g["share"] = (g.ext / g.tot).clip(0, 1)
    g["funding"] = g.tot / g.ny                       # programme funding a year, all sources (US$, average of the years)
    g["years"] = [f"{a}" if a == b else f"{a}-{b}" for a, b in zip(g.y0, g.y1)]
    return g


def programme_share(iso_all: list, region: pd.Series) -> pd.DataFrame:
    ex = pd.read_csv(WHO_TB / "TB_expenditure_utilisation_2026-10-14.csv")
    bu = pd.read_csv(WHO_TB / "TB_budget_2026-10-14.csv")
    rc, cf = _pooled(ex, "rcvd", range(2021, 2025)), _pooled(bu, "cf", range(2022, 2026))
    t = pd.DataFrame(index=pd.Index(iso_all, name="iso3"))
    t["who_region"] = region.reindex(iso_all)
    t["tb_prog_ext_share"], t["tb_prog_ext_year"] = rc.share.reindex(iso_all), rc.years.reindex(iso_all)
    t["tb_prog_funding_usd"] = rc.funding.reindex(iso_all)
    t["tb_prog_ext_source"] = np.where(t.tb_prog_ext_share.notna(), "WHO TB funding received", "")
    m = t.tb_prog_ext_share.isna() & cf.share.reindex(iso_all).notna()
    t.loc[m, "tb_prog_ext_share"] = cf.share.reindex(iso_all)[m]
    t.loc[m, "tb_prog_ext_year"] = cf.years.reindex(iso_all)[m]
    t.loc[m, "tb_prog_funding_usd"] = cf.funding.reindex(iso_all)[m]
    t.loc[m, "tb_prog_ext_source"] = "WHO TB expected funding (budget file)"
    # received funding is incompletely reported when the domestic amount is copied forward (the same figure in each
    # pooled year) and the total received is below the expected funding for the same years (Iraq: US$0.84M domestic in
    # each of 2022-2024 against US$19.5M expected; South Africa: US$51.43M in 2023 and 2024 against US$120-168M):
    # expected funding instead. Received below expected on its own is common (pledges not all paid) and is kept
    exs = _screen(ex, "rcvd")
    bus = _screen(bu, "cf").set_index(["iso3", "year"]).cf_tot_sources
    m = pd.Series(False, index=t.index)
    for iso in t.index[(t.tb_prog_ext_source == "WHO TB funding received")]:
        y0, y1 = (int(x) for x in (str(t.loc[iso, "tb_prog_ext_year"]).split("-") * 2)[:2])
        r = exs[(exs.iso3 == iso) & exs.year.between(y0, y1) & (exs.rcvd_tot_domestic > 0)]
        exp_ = bus.reindex([(iso, y) for y in r.year]).sum()
        m[iso] = len(r) >= 2 and r.rcvd_tot_domestic.nunique() == 1 and r.rcvd_tot_sources.sum() < exp_
    t.loc[m, "tb_prog_ext_share"] = cf.share.reindex(iso_all)[m]
    t.loc[m, "tb_prog_ext_year"] = cf.years.reindex(iso_all)[m]
    t.loc[m, "tb_prog_funding_usd"] = cf.funding.reindex(iso_all)[m]
    t.loc[m, "tb_prog_ext_source"] = "WHO TB expected funding (received funding incompletely reported)"
    t["tb_prog_ext_share_expected"] = cf.share.reindex(iso_all)       # comparison: the share in the budget file
    t["tb_prog_ext_year_expected"] = cf.years.reindex(iso_all)
    # no year with domestic funding above zero: the latest received year 2021-2024 only where it reports domestic
    # funding as zero (Somalia: 0 in every year); a blank domestic entry means not reported, not zero, so those
    # countries (and any whose latest data are older than 2021) take the regional median below
    e = _screen(ex, "rcvd")
    e = e[e.year.between(2021, 2024) & (e.rcvd_tot_domestic == 0)]
    e["sh"] = (e[[f"rcvd_tot_{k}" for k in EXT]].sum(axis=1, min_count=1).fillna(0.0) / e.rcvd_tot_sources).clip(0, 1)
    la = e.sort_values("year").groupby("iso3").tail(1).set_index("iso3")
    m = t.tb_prog_ext_share.isna() & la.sh.reindex(iso_all).notna()
    t.loc[m, "tb_prog_ext_share"] = la.sh.reindex(iso_all)[m]
    t.loc[m, "tb_prog_ext_year"] = la.year.reindex(iso_all)[m].astype(int).astype(str)
    t.loc[m, "tb_prog_funding_usd"] = la.rcvd_tot_sources.reindex(iso_all)[m]
    t.loc[m, "tb_prog_ext_source"] = "WHO TB funding received (domestic funding reported as zero)"
    have = t[t.tb_prog_ext_share.notna()]
    for iso in t.index[t.tb_prog_ext_share.isna()]:
        pool = have[have.who_region == t.loc[iso, "who_region"]]
        label = f"regional median ({t.loc[iso, 'who_region']}, n={len(pool)})"
        if len(pool) < 3:
            pool, label = have, f"median of model countries (n={len(have)})"
        t.loc[iso, "tb_prog_ext_share"] = float(pool.tb_prog_ext_share.median())
        t.loc[iso, "tb_prog_ext_source"] = label
    # comparisons: the single-year value the model used before (batch34_donor_share.csv) and Clark et al.'s 2023 value
    old = pd.read_csv(ROOT / "audit" / "logic_review" / "batch34_donor_share.csv").set_index("iso3")
    t["donor_share_tb_before"] = old.TB_share.reindex(iso_all)
    clark = pd.read_csv(LSHTM / "funding_scenario_values.csv").set_index("code")
    t["clark_usaid_gf_share_2023"] = clark.scen_usaid_gf.reindex(iso_all)
    # WHO estimated TB deaths a year, all forms (HIV-negative + HIV-positive), latest year, with WHO's 95% interval
    b = pd.read_csv(WHO_TB / "TB_burden_countries_2026-10-14.csv")
    b = b[b.e_mort_num.notna()].sort_values("year").groupby("iso3").tail(1).set_index("iso3")
    for c, k in (("tb_deaths_who", "e_mort_num"), ("tb_deaths_who_lo", "e_mort_num_lo"),
                 ("tb_deaths_who_hi", "e_mort_num_hi"), ("tb_deaths_who_year", "year")):
        t[c] = b[k].reindex(iso_all)
    return t


def fit_curve() -> tuple:
    f = pd.read_csv(LSHTM / "funding_scenario_values.csv").set_index("code")
    c = pd.read_csv(LSHTM / "countries.csv").set_index("CountryCode")
    b = pd.read_csv(WHO_TB / "TB_burden_countries_2026-10-14.csv")
    b = b[b.year == 2023].set_index("iso3")
    d = f.join(c[["Income", "WHOregion"]]).join(b[["e_mort_num"]])
    obs = np.array([p[3] for p in PUBLISHED]) / 100
    se = np.array([(p[5] - p[4]) / 3.92 for p in PUBLISHED]) / 100
    groups = [d if p[2] == "All" else d[(d.Income == p[2]) | (d.WHOregion == p[2])] for p in PUBLISHED]
    a, bq = 0.5, 0.5
    for _ in range(50):           # the denominator (1 + g(u)) uses the previous (a, b): converges in a few rounds
        X = []
        for p, m in zip(PUBLISHED, groups):
            w, x = m.e_mort_num.to_numpy(), m[p[0]].to_numpy()
            if p[1] is None:
                X.append([(w * x).sum() / w.sum(), (w * x ** 2).sum() / w.sum()])
            else:
                u = m[p[1]].to_numpy()
                den = (w * (1 + a * u + bq * u ** 2)).sum()
                X.append([(w * (x - u)).sum() / den, (w * (x ** 2 - u ** 2)).sum() / den])
        X = np.array(X)
        new = np.linalg.lstsq(X / se[:, None], obs / se, rcond=None)[0]
        if np.allclose(new, (a, bq), atol=1e-10):
            break
        a, bq = new
    pred = X @ np.array([a, bq])
    rows = pd.DataFrame({"scenario": [p[0] for p in PUBLISHED], "relative_to": [p[1] or "status quo" for p in PUBLISHED],
                         "group": [p[2] for p in PUBLISHED], "n_countries": [len(m) for m in groups],
                         "published_pct": obs * 100, "published_lo": [p[4] for p in PUBLISHED],
                         "published_hi": [p[5] for p in PUBLISHED], "fitted_pct": pred * 100})
    rows["within_ui"] = (rows.fitted_pct >= rows.published_lo) & (rows.fitted_pct <= rows.published_hi)
    chi2 = float((((pred - obs) / se) ** 2).sum())
    return a, bq, chi2, rows


def main():
    lines = pd.read_csv(ROOT / "model_data" / "dah_lines.csv")
    iso_all = sorted(set(lines.iso3))
    ci = pd.read_csv(ROOT / "model_data" / "country_inputs.csv").set_index("iso3")
    t = programme_share(iso_all, ci.who_region)
    t.reset_index().to_csv(OUT_SHARE, index=False)
    a, bq, chi2, rows = fit_curve()
    rows.round(3).to_csv(OUT_FIT, index=False)
    pd.set_option("display.width", 250)
    print(f"fit: rise in TB deaths = {a:.3f} f + {bq:.3f} f^2   (chi2 {chi2:.1f} on {len(rows) - 2} df; "
          f"{int(rows.within_ui.sum())} of {len(rows)} fitted values inside the published 95% interval)")
    print("g(f) at f = 0.1, 0.25, 0.5, 0.75, 1:", [round(a * x + bq * x * x, 3) for x in (0.1, 0.25, 0.5, 0.75, 1.0)])
    print(rows.round(1).to_string(index=False))
    print(t.tb_prog_ext_source.str.split(" \\(").str[0].value_counts().to_string())
    dah = lines[lines.bucket == "TB"].groupby("iso3").usd.sum()
    t["full_cut_share_lost"] = np.minimum(t.tb_prog_ext_share, (dah.reindex(t.index) / t.tb_prog_funding_usd).fillna(9))
    print(t[["tb_prog_ext_share", "tb_prog_ext_year", "tb_prog_funding_usd", "full_cut_share_lost",
             "donor_share_tb_before", "clark_usaid_gf_share_2023"]]
          .round(2).to_string())


if __name__ == "__main__":
    main()
