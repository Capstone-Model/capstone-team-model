"""TB redesign review (Nov 2026): every country under a full cut of donor TB money, both aid sources. Run from the repo root:
    python audit/tb_review/country_review.py audit/tb_review/country_review.csv
Columns: model vs WHO deaths, incidence and detection; external share and programme funding; aid vs WHO external money;
share of the programme lost (f_full); rise in TB deaths a year; extra deaths per person not starting treatment vs the
case-fatality gap (must stay below); the US-bilateral share lost vs the USAID share Clark et al. used."""
import sys, logging
sys.path.insert(0, "."); logging.disable(logging.WARNING)
import numpy as np, pandas as pd
import health_model as hm
W = "datasets/WHO_TB/"
bur = pd.read_csv(W + "TB_burden_countries_2026-10-14.csv"); bur = bur[bur.year == bur.year.max()].set_index("iso3")
noti = pd.read_csv(W + "TB_notifications_2026-10-14.csv", usecols=["iso3", "year", "c_newinc"])
noti = noti[noti.year == noti.year.max()].set_index("iso3")
clark = pd.read_csv("datasets/LSHTM_TB/funding_scenario_values.csv").set_index("code")
P = hm.load_params(); Pc = hm.draw_params(P, 0)
us_bi = hm.Scenario(name="US bilateral TB", bucket_only="TB"); us_bi.src_direct = {"United States": 1.0}
out = []
import inspect
SOURCES = ("ihme", "oecd") if "source" in inspect.signature(hm.load_inputs).parameters else (None,)
for src in SOURCES:                    # both aid data sources where the engine has the toggle, else the default
    I = hm.load_inputs(source=src) if src else hm.load_inputs(); ci = I["ci"]; L = I["lines"]
    src = src or "ihme"
    for iso in sorted(set(L.iso3)):
        tb = L[(L.iso3 == iso) & (L.bucket == "TB")].usd.sum()
        if tb <= 0 or iso not in ci.index: continue
        row = ci.loc[iso]
        r = hm.run_country(iso, hm.uniform_scenario("TB", 1.0), hm.Fiscal(), I, P, n_draws=0)
        ln = r["lines"].set_index("line")
        starts = ln.loc["tb_ds", "need"] * ln.loc["tb_ds", "cov0"]
        lost_starts = ln.loc["tb_ds", "units_lost"]
        f = lost_starts / starts if starts > 0 else np.nan
        D = hm.tb_deaths_base(row)
        tot = r["buckets"].set_index("bucket").loc["TB", "deaths_5y"] + r["committed"]["after_parts"]["tb"]
        dcfr = float(hm.per_unit_levels(row, Pc, hm.epi(row, Pc)["art_cov"])["tb_ds"][0, 0])
        rec = {"src": src, "iso3": iso, "D_model": D, "D_who": bur.e_mort_num.get(iso), "inc_model": ln.loc["tb_ds", "need"],
               "inc_who": bur.e_inc_num.get(iso), "cdr_model": ln.loc["tb_ds", "cov0"], "cdr_who": bur.c_cdr.get(iso, np.nan) / 100,
               "notified_who": noti.c_newinc.get(iso), "hiv_sh": row.get("tb_hiv_share"),
               "ext": row.tb_prog_ext_share, "ext_src": row.tb_prog_ext_source, "ext_yr": row.tb_prog_ext_year,
               "prog_m": row.tb_prog_funding_usd / 1e6, "aid_m": tb / 1e6,
               "aid_vs_who_ext": tb / (row.tb_prog_ext_share * row.tb_prog_funding_usd) if row.tb_prog_funding_usd > 0 else np.nan,
               "f_full": f, "rise_pct_yr": 100 * tot / 5 / D, "extra_per_yr": tot / 5,
               "deaths_per_lost_start": tot / 5 / lost_starts if lost_starts > 0 else np.nan, "dcfr": dcfr,
               "clark_usaid_gf": clark.scen_usaid_gf.get(iso), "clark_usaid": clark.scen_usaid.get(iso),
               "budget_per_notified": row.tb_prog_funding_usd / noti.c_newinc.get(iso, np.nan)}
        ru = hm.run_country(iso, us_bi, hm.Fiscal(), I, P, n_draws=0).get("lines").set_index("line")
        rec["f_us_bilateral"] = ru.loc["tb_ds", "units_lost"] / starts / (1 - Pc["continuity"][0]) if starts > 0 else np.nan
        out.append(rec)
t = pd.DataFrame(out)
t.to_csv(sys.argv[1], index=False)
