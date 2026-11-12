"""
Allocation model: dividing the same post-cut health aid pool by ability to replace aid, cost per life saved and need.

The method is the team workbook "ALLOCATION MODEL/Allocation_model_CostPerLife_Need.xlsx" (its explanation and results
are the two PDFs in that folder). This file rebuilds the workbook's formulas in Python from the workbook's raw data
sheets (Data_Aid, Data_Tax, Data_KS, Data_HealthHist, Data_Need and the Assumptions tab), recomputes the cost per life
saved with this repo's engine (health_model.run_country, as the workbook's Data_CostPerLife tab was made), and runs the
four scenarios through the engine. The Allocation page (views/5_Allocation.py) reads what it writes.

    python allocation_model.py          # writes model_data/allocation/*.csv (about a minute)
    python allocation_model.py --check  # also compares the workbook's own cost-per-life values with the engine's

The chain, per country i (US$ millions per year):
    capacity_i   = new tax revenue x health share of government spending
                   + room factor (Kharas & Sarkar score) x own best 4-year rise in health share x gov. spending x 0.625
    need disc.   = 1 / (1 + k_need x need_i)                 need: DAH-adjusted need index (team disease-burden model)
    eff. disc.   = 1 / (1 + k_eff x efficiency_i)            efficiency: lives saved per US$1M / the highest country
    effective_i  = capacity_i x need disc. x eff. disc.
    cut_i        = min(aid2023_i, lambda x effective_i)       lambda solved so the cuts add up to the OECD-projected cut
Four scenarios run through the engine for every country in both the workbook and the dashboard:
    Red    current split (OECD 2026 projection), no new money      cut rate = projected cut / 2023 aid, theta = 0
    Yellow current split + new money                               theta = min(capacity, projected cut) / projected cut
    Green  reallocated + new money                                 cut rate = model cut / 2023 aid, theta = min(cap, cut)/cut
    Blue   reallocated, no new money                               theta = 0
Each run: health_model.run_country, the cut rate on every donor / channel / service cell, Fiscal(custom, theta,
cap_to_space=False, pro_rata, p75), 400 draws, seed 7. Four allocation rules are run (capacity only, + need,
+ cost per life, + both); Red and Yellow do not depend on the rule.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

HERE = Path(__file__).resolve().parent
WORKBOOK = HERE / "ALLOCATION MODEL" / "Allocation_model_CostPerLife_Need.xlsx"
OUT = HERE / "model_data" / "allocation"

N_DRAWS, SEED = 400, 7
BUCKETS = ["HIV", "TB", "Malaria", "Immunization"]
# the eight services the results report (HIV prevention and orphan care have no coverage measure)
SERVICES = ["hiv_art", "hiv_pmtct", "tb_ds", "tb_dr", "mal_itn", "mal_irs", "mal_cm", "imm"]
# allocation rules: key -> (label, need k, efficiency k); the workbook's levers are Assumptions B26 and B37
RULES = {"cpl_need": ("Ability to pay + cost per life + need", None, None),     # None = the workbook's own k
         "cpl": ("Ability to pay + cost per life", 0.0, None),
         "need": ("Ability to pay + need", None, 0.0),
         "cap": ("Ability to pay only", 0.0, 0.0)}
PLAN = "cpl_need"
SCENARIOS = {"red": "Current split, no new money", "yellow": "Current split + new money",
             "green": "Reallocated + new money", "blue": "Reallocated, no new money"}


# --------------------------------------------------------------------------- #
# Reading the workbook's raw data (typed values only: no formula is read)
# --------------------------------------------------------------------------- #
def _rows(ws, first: int, last: int, cols: str) -> list:
    out = []
    for r in range(first, last + 1):
        out.append([ws[f"{c}{r}"].value for c in cols.split()])
    return out


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else np.nan


def read_workbook(path: Path = WORKBOOK) -> dict:
    """The workbook's data sheets and levers. Raw data rows are typed values; the few levers that are formulas
    (Assumptions B7, B8) are recomputed here exactly as the workbook writes them."""
    wb = openpyxl.load_workbook(path, data_only=False)
    A = wb["Assumptions"]
    lev = {k: A[k].value for k in ("B6", "B10", "B11", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20", "B21",
                                   "B22", "B23", "B25", "B26", "B28", "B33", "B37")}
    lev["B7"] = 2.5 if lev["B6"] == "Average 2027-30" else float(lev["B6"]) - 2026
    lev["B8"] = 1.0 if lev["B15"] == "Immediate" else lev["B7"] / 4
    for k in ("B11", "B14", "B17", "B19", "B20", "B21", "B22", "B23", "B26", "B33", "B37"):
        lev[k] = float(lev[k])
    expected = {"B6": "Average 2027-30", "B10": "Country health share", "B13": "Own best 4-yr gain",
                "B15": "Phased 2027-30", "B16": "No", "B18": "Score", "B25": "DAH-adjusted need", "B28": "No"}
    bad = {k: (lev[k], v) for k, v in expected.items() if lev[k] != v}
    if lev["B33"] != 0:
        bad["B33"] = (lev["B33"], 0)                     # the equal-% blend is not implemented
    if bad:     # the Python chain implements these settings (the workbook's defaults); others would need code here
        raise ValueError(f"workbook levers differ from the settings this file implements: {bad}")

    aid = pd.DataFrame(_rows(wb["Data_Aid"], 5, 86, "A B C D E F G"),
                       columns=["iso3", "country", "income", "dah_2023", "dah_2026", "gov_spend", "health_pct"])
    tax = pd.DataFrame(_rows(wb["Data_Tax"], 5, 86, "A C D E F G H I J"),
                       columns=["iso3", "in_proj", "t2027", "t2028", "t2029", "t2030", "flag", "fb_gain", "fb_pct"])
    ks = pd.DataFrame(_rows(wb["Data_KS"], 5, 148, "A C D"), columns=["iso3", "ks_score", "ks_cluster"])
    yrs = [chr(c) for c in range(ord("C"), ord("Z") + 1)]                 # C..Z = 2000..2023
    hh = pd.DataFrame(_rows(wb["Data_HealthHist"], 5, 85, "A " + " ".join(yrs)),
                      columns=["iso3"] + list(range(2000, 2024)))
    need = pd.DataFrame(_rows(wb["Data_Need"], 5, 75, "A F H"), columns=["iso3", "composite", "dah_adj"])
    tiers = pd.DataFrame(_rows(wb["Data_Need"], 5, 200, "K L"), columns=["iso3", "tier"]).dropna(subset=["iso3"])
    cpl = pd.DataFrame(_rows(wb["Data_CostPerLife"], 5, 86, "A D E F G H I J K L"),
                       columns=["iso3", "in_dash", "s_HIV", "s_TB", "s_Malaria", "s_Immunization",
                                "l_HIV", "l_TB", "l_Malaria", "l_Immunization"])
    for d in (aid, tax, ks, hh, need, cpl):
        d.dropna(subset=["iso3"], inplace=True)
    for y in range(2000, 2024):
        hh[y] = hh[y].map(_num)
    return {"lev": lev, "aid": aid.set_index("iso3"), "tax": tax.set_index("iso3"), "ks": ks.set_index("iso3"),
            "hh": hh.set_index("iso3"), "need": need.set_index("iso3"), "tiers": tiers, "cpl_wb": cpl.set_index("iso3")}


# --------------------------------------------------------------------------- #
# Steps 1-5: domestic capacity (Capacity tab)
# --------------------------------------------------------------------------- #
def best_four_year_gain(hh: pd.DataFrame) -> pd.Series:
    """Data_HealthHist AU: the largest rise in health's share of government spending over any 4-year window
    2000-2004 ... 2019-2023 (percentage points / 100), floored at 0; blank when no window has both ends."""
    gains = pd.concat({y: hh[y + 4] - hh[y] for y in range(2000, 2020)}, axis=1)
    best = gains.max(axis=1, skipna=True).clip(lower=0) / 100
    return best.where(gains.notna().any(axis=1))


def capacity(W: dict) -> pd.DataFrame:
    lev, aid, tax, ks = W["lev"], W["aid"], W["tax"], W["ks"]
    df = aid[["country", "income"]].copy()
    df["aid_2023"] = aid["dah_2023"].map(_num) / 1e6
    df["aid_2026_proj"] = aid["dah_2026"].map(_num) / 1e6
    t = tax.reindex(df.index)
    in_proj = t["in_proj"] == 1
    # E: tax workbook, average of the 2027-30 additions (US$ bn -> m)
    proj = t[["t2027", "t2028", "t2029", "t2030"]].astype(float).mean(axis=1) * 1000
    # H: fallback, recommended max gain x average over 1-4 years of min(1, pace x years / gain as % of GDP)
    g = t["fb_pct"].map(_num)
    ramp = sum(np.minimum(1.0, lev["B11"] * k / g) for k in (1, 2, 3, 4)) / 4
    fb = (t["fb_gain"].map(_num).fillna(0) / 1e6) * ramp.where(g > 0, 0.0)
    df["tax_source"] = np.where(in_proj, "Tax projection", "Fallback")
    df["new_tax"] = np.where(in_proj, proj, fb)
    df["health_share"] = aid["health_pct"].map(_num) / 100
    df["tax_for_health"] = df["new_tax"] * df["health_share"].fillna(0)
    df["gov_spend"] = aid["gov_spend"].map(_num) / 1e6
    k = ks.reindex(df.index)
    df["ks_score"] = k["ks_score"].map(_num)
    df["ks_cluster"] = k["ks_cluster"].fillna("none")
    room = ((df["ks_score"] - lev["B19"]) / (lev["B20"] - lev["B19"])).clip(0, 1)
    df["room"] = room.where(df["ks_score"].notna(), lev["B23"])
    best = best_four_year_gain(W["hh"])
    med = float(best.median())
    df["shift_cap"] = best.reindex(df.index).fillna(med)
    df["shift_cap_source"] = np.where(best.reindex(df.index).notna(), "own history", "median")
    df["shift_cap"] = df["shift_cap"].where(df["health_share"].notna(), 0.0)       # no health-share data: no shift
    df["budget_shift"] = df["room"] * df["shift_cap"] * df["gov_spend"].fillna(0) * lev["B8"]
    df["capacity"] = df["tax_for_health"] + df["budget_shift"]
    return df


# --------------------------------------------------------------------------- #
# Step 6: need (Need tab)
# --------------------------------------------------------------------------- #
def need(W: dict, index: pd.Index) -> pd.DataFrame:
    col = "dah_adj" if W["lev"]["B25"] == "DAH-adjusted need" else "composite"
    nd = W["need"][col].map(_num)
    tier = W["tiers"].drop_duplicates("iso3").set_index("iso3")["tier"]
    med_all = float(nd.median())
    med_hi = float(nd[tier.reindex(nd.index) == "High"].median())
    med_lo = float(nd[tier.reindex(nd.index) == "Low"].median())
    out = pd.DataFrame(index=index)
    out["need_model"] = nd.reindex(index)
    t = tier.reindex(index).fillna("none")
    fb = np.where(t == "High", med_hi, np.where(t == "Low", med_lo, med_all))
    out["need"] = out["need_model"].fillna(pd.Series(fb, index=index))
    out["need_source"] = np.where(out["need_model"].notna(), "Disease-burden model",
                                  np.where(t == "none", "Median of all", "Median of " + t + "-need tier"))
    out["need_high"] = out["need"] >= med_all
    return out


# --------------------------------------------------------------------------- #
# Step 8: cost per life saved, from the engine (Data_CostPerLife), and efficiency (Efficiency tab)
# --------------------------------------------------------------------------- #
def cost_per_life(isos, inputs, ptab) -> pd.DataFrame:
    """Every HIV, TB, malaria and immunization aid line cut by 10%, no government replacement, central values:
    lives per US$1M = extra deaths 2026-2030 per US$1M of yearly aid lost, per disease; shares = each disease's share
    of the aid lost (= of its aid). The same run as the workbook's Data_CostPerLife tab."""
    import health_model as hm
    rows = {}
    for iso in isos:
        sc = hm.Scenario(name="10% cut", src_direct={s: 0.1 for s in hm.SOURCE_GROUPS},
                         src_multi={s: 0.1 for s in hm.SOURCE_GROUPS})
        b = hm.run_country(iso, sc, hm.Fiscal(), inputs, ptab, n_draws=0)["buckets"].set_index("bucket")
        lost = b["gross_loss_usd"].reindex(BUCKETS).fillna(0.0)
        d = b["deaths_5y"].reindex(BUCKETS).fillna(0.0)
        r = {f"s_{k}": lost[k] / lost.sum() for k in BUCKETS}
        r.update({f"l_{k}": (d[k] / (lost[k] / 1e6)) if lost[k] > 0 else np.nan for k in BUCKETS})
        rows[iso] = r
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["lives_per_musd"] = sum(out[f"s_{k}"] * out[f"l_{k}"].fillna(0) for k in BUCKETS)
    out["cost_per_life"] = 1e6 / out["lives_per_musd"]
    return out


def efficiency(cap: pd.DataFrame, cpl: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=cap.index)
    out["lives_per_musd"] = cpl["lives_per_musd"].reindex(cap.index)
    out["cost_per_life"] = 1e6 / out["lives_per_musd"]
    out["eff_index"] = out["lives_per_musd"] / out["lives_per_musd"].max()
    grp = out.groupby(cap["income"])["eff_index"].transform("mean")        # AVERAGEIFS by income group
    out["in_dashboard"] = out["eff_index"].notna()
    out["efficiency"] = out["eff_index"].fillna(grp)
    out["eff_source"] = np.where(out["in_dashboard"], "Dashboard", "Income-group average")
    out["eff_high"] = out["efficiency"] >= out["efficiency"].median()
    return out


# --------------------------------------------------------------------------- #
# Step 9: the cut solver (Cut_Solver tab)
# --------------------------------------------------------------------------- #
def solve_cuts(aid_2023: pd.Series, effcap: pd.Series, total_cut: float) -> tuple:
    """cut_i = min(aid_i, lambda x effcap_i) with sum = total_cut. Countries whose breakpoint aid_i / effcap_i is at or
    below the largest feasible breakpoint lose all their aid; lambda spreads the rest over the others. Same as the
    workbook's breakpoint method (no Solver)."""
    a, e = aid_2023.to_numpy(float), effcap.to_numpy(float)
    bp = np.where(e > 0, a / np.where(e > 0, e, 1), 1e99)
    tot = np.array([a[bp <= b].sum() + (0 if b >= 1e98 else b * e[bp > b].sum()) for b in bp])
    feas = bp[tot <= total_cut + 1e-6]
    m8 = feas.max() if len(feas) else 0.0
    lam = (total_cut - a[bp <= m8].sum()) / e[bp > m8].sum()
    cut = np.where(bp <= m8, a, lam * e)
    assert abs(cut.sum() - total_cut) < 1e-6, (cut.sum(), total_cut)
    return pd.Series(cut, index=aid_2023.index), float(lam)


def allocate(base: pd.DataFrame, need_k: float, eff_k: float) -> tuple:
    """Model cut and 2026 aid for one pair of k values. base: capacity, need and efficiency columns."""
    nd = 1 / (1 + need_k * base["need"])
    ed = 1 / (1 + eff_k * base["efficiency"])
    eff_cap = base["capacity"] * nd * ed
    total_cut = base["aid_2023"].sum() - base["aid_2026_proj"].sum()
    cut, lam = solve_cuts(base["aid_2023"], eff_cap, total_cut)
    return pd.DataFrame({"need_disc": nd, "eff_disc": ed, "eff_capacity": eff_cap, "cut": cut,
                         "aid_2026": base["aid_2023"] - cut}), lam


def build_base(W: dict, cpl: pd.DataFrame) -> pd.DataFrame:
    cap = capacity(W)
    return pd.concat([cap, need(W, cap.index), efficiency(cap, cpl)], axis=1)


def need_tiers(n: pd.Series) -> pd.Series:
    """High / Medium / Low disease need: thirds of the 82 countries by the need index used (pandas qcut, as in the
    funding-hole chart). Countries tied at a fallback value fall in the same third, so the thirds are 27 / 25 / 30."""
    return pd.qcut(n, 3, labels=["Low", "Medium", "High"]).astype(str)


# --------------------------------------------------------------------------- #
# The engine runs
# --------------------------------------------------------------------------- #
def run_scenario(iso, cut_rate, theta, inputs, ptab):
    import health_model as hm
    sc = hm.Scenario(name="allocation", src_direct={s: cut_rate for s in hm.SOURCE_GROUPS},
                     src_multi={s: cut_rate for s in hm.SOURCE_GROUPS})
    fis = hm.Fiscal(mode="custom", theta=theta, cap_to_space=False, allocation="pro_rata", effort="p75")
    return hm.run_country(iso, sc, fis, inputs, ptab, n_draws=N_DRAWS, seed=SEED, keep_draws=True)


def run_all(base: pd.DataFrame, allocs: dict, inputs, ptab) -> dict:
    """Every scenario for every matched country. Returns long tables: by country x disease, by country x service,
    and the draw-level totals."""
    matched = [i for i in base.index if i in set(inputs["lines"].iso3)]
    by_dis, by_line, draws = [], [], []
    runs = [("base", "red"), ("base", "yellow")] + [(r, s) for r in allocs for s in ("blue", "green")]
    for rule, scen in runs:
        cut = base["aid_2023"] - base["aid_2026_proj"] if rule == "base" else allocs[rule]["cut"]
        for iso in matched:
            e, c, cap = base.at[iso, "aid_2023"], cut[iso], base.at[iso, "capacity"]
            rate = c / e if e > 0 else 0.0
            theta = 0.0 if scen in ("red", "blue") else (min(cap, c) / c if c > 0 else 0.0)
            r = run_scenario(iso, rate, theta, inputs, ptab)
            b = r["buckets"].set_index("bucket").reindex(BUCKETS)
            for k in BUCKETS:
                by_dis.append({"rule": rule, "scenario": scen, "iso3": iso, "bucket": k, "cut_rate": rate,
                               "theta": theta, "base_usd": b.at[k, "base_usd"], "gross_usd": b.at[k, "gross_loss_usd"],
                               "net_usd": b.at[k, "net_loss_usd"], "deaths_5y": b.at[k, "deaths_5y"]})
            L = r["lines"].set_index("line").reindex(SERVICES)
            for ln in SERVICES:
                by_line.append({"rule": rule, "scenario": scen, "iso3": iso, "line": ln,
                                "label": L.at[ln, "label"], "unit_label": L.at[ln, "unit_label"],
                                "need": L.at[ln, "need"], "cov0": L.at[ln, "cov0"],
                                "cov_drop_pp": L.at[ln, "cov_drop_pp"], "units_lost": L.at[ln, "units_lost"]})
            d = np.asarray(r["draw_totals"]["deaths_5y"], dtype=float)
            draws.append(pd.DataFrame({"rule": rule, "scenario": scen, "iso3": iso, "draw": np.arange(len(d)),
                                       "deaths_5y": d}))
    D = pd.concat(draws).groupby(["rule", "scenario", "draw"], as_index=False)["deaths_5y"].sum()
    return {"matched": matched, "by_disease": pd.DataFrame(by_dis), "by_line": pd.DataFrame(by_line), "draws": D}


# --------------------------------------------------------------------------- #
# Reading the results back (the Allocation page and the tests use these)
# --------------------------------------------------------------------------- #
def load(out: Path = OUT) -> dict:
    return {"countries": pd.read_csv(out / "countries.csv", index_col="iso3", keep_default_na=False,
                                     na_values=[""]),
            "cost_per_life": pd.read_csv(out / "cost_per_life.csv", index_col="iso3", keep_default_na=False,
                                         na_values=[""]),
            "rules": pd.read_csv(out / "rules.csv").set_index("rule"),
            "disease": pd.read_csv(out / "results_disease.csv"),
            "services": pd.read_csv(out / "results_services.csv"),
            "draws": pd.read_csv(out / "results_draws.csv")}


def _pick(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Rows for the four scenarios under one rule: Red and Yellow are shared by every rule ('base')."""
    return df[((df["rule"] == "base") & df["scenario"].isin(["red", "yellow"]))
              | ((df["rule"] == rule) & df["scenario"].isin(["blue", "green"]))]


def deaths_by_disease(R: dict, rule: str = PLAN) -> pd.DataFrame:
    """Extra deaths 2026-2030, disease x scenario (central), with a Total row."""
    t = _pick(R["disease"], rule).pivot_table(index="bucket", columns="scenario", values="deaths_5y", aggfunc="sum")
    t = t.reindex(index=BUCKETS, columns=list(SCENARIOS))
    t.loc["Total"] = t.sum()
    return t


def deaths_range(R: dict, rule: str, first: str | None, second: str, q=(2.5, 97.5)) -> tuple:
    """95% range of deaths in `second` (first=None) or of lives saved first -> second, draw by draw."""
    d = _pick(R["draws"], rule).pivot_table(index="draw", columns="scenario", values="deaths_5y")
    x = d[second] if first is None else d[first] - d[second]
    return tuple(float(v) for v in np.percentile(x, q))


def rule_gain_range(R: dict, rule_a: str, rule_b: str, scenario: str = "blue", q=(2.5, 97.5)) -> tuple:
    """95% range of lives saved by rule_b instead of rule_a in one scenario, draw by draw."""
    d = R["draws"][R["draws"]["scenario"] == scenario].pivot_table(index="draw", columns="rule", values="deaths_5y")
    return tuple(float(v) for v in np.percentile(d[rule_a] - d[rule_b], q))


def coverage(R: dict, rule: str = PLAN) -> pd.DataFrame:
    """Coverage of each service by scenario: baseline coverage minus the modelled drop, weighted by each country's
    population in need of the service, over the matched countries that report it."""
    s = _pick(R["services"], rule).copy()
    s = s[(s["need"] > 0) & s["cov0"].notna()]
    s["cov"] = s["cov0"] - s["cov_drop_pp"].fillna(0) / 100
    s["w"] = s["need"] * s["cov"]
    g = s.groupby(["line", "scenario"])
    t = (g["w"].sum() / g["need"].sum()).unstack("scenario")
    return t.reindex(index=SERVICES, columns=list(SCENARIOS))


def units_lost(R: dict, rule: str = PLAN) -> pd.DataFrame:
    """Service units lost a year (average 2026-2030), service x scenario."""
    t = _pick(R["services"], rule).pivot_table(index="line", columns="scenario", values="units_lost", aggfunc="sum")
    return t.reindex(index=SERVICES, columns=list(SCENARIOS)).fillna(0.0)


def disease_medians(R: dict, min_share: float = 0.05) -> pd.DataFrame:
    """Median lives saved per US$1M and cost per life saved, by disease, over the dashboard countries where the
    disease is at least `min_share` of their HIV/TB/malaria/immunization aid (what drives the efficiency ranking)."""
    c = R["cost_per_life"]
    rows = {}
    for k in BUCKETS:
        v = c.loc[c[f"s_{k}"] >= min_share, f"l_{k}"].dropna()
        rows[k] = {"lives_per_musd": v.median(), "cost_per_life": (1e6 / v).median(), "countries": len(v)}
    return pd.DataFrame(rows).T


def country_results(R: dict, rule: str = PLAN) -> pd.DataFrame:
    """Per matched country: deaths and aid lost after replacement in each scenario (central)."""
    d = _pick(R["disease"], rule)
    deaths = d.pivot_table(index="iso3", columns="scenario", values="deaths_5y", aggfunc="sum")
    net = d.pivot_table(index="iso3", columns="scenario", values="net_usd", aggfunc="sum")
    return deaths.add_prefix("deaths_").join(net.add_prefix("net_usd_"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="compare the workbook's cost per life with the engine's")
    args = p.parse_args(argv)
    import health_model as hm
    inputs, ptab = hm.load_inputs(), hm.load_params()
    W = read_workbook()
    engine = sorted(set(inputs["lines"].iso3))
    cpl_all = cost_per_life(engine, inputs, ptab)          # all dashboard countries (the per-disease medians)
    cpl = cpl_all.reindex([i for i in W["aid"].index if i in engine])
    if args.check:
        wb = W["cpl_wb"].reindex(cpl.index)
        for k in BUCKETS:
            rel = (cpl[f"l_{k}"] / wb[f"l_{k}"].map(_num) - 1).abs()
            moved = rel[rel > 1e-6]
            print(f"{k:13s} lives per US$1M: {len(moved)} of {rel.notna().sum()} countries differ from the workbook"
                  + (f" (largest {moved.max():.1%}: {', '.join(moved.sort_values().index[-5:])})" if len(moved) else ""))
    base = build_base(W, cpl)
    base["need_tier"] = need_tiers(base["need"])
    lev = W["lev"]
    allocs, summary = {}, []
    for key, (label, nk, ek) in RULES.items():
        nk = lev["B26"] if nk is None else nk
        ek = lev["B37"] if ek is None else ek
        a, lam = allocate(base, nk, ek)
        allocs[key] = a
        summary.append({"rule": key, "label": label, "need_k": nk, "eff_k": ek, "lambda": lam,
                        "zero_aid": int((a["aid_2026"] <= 1e-4).sum()),
                        "gain_vs_proj": int((a["aid_2026"] - base["aid_2026_proj"] >= 0.05).sum()),
                        "lose_vs_proj": int((a["aid_2026"] - base["aid_2026_proj"] <= -0.05).sum())})
    R = run_all(base, allocs, inputs, ptab)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["country", "income", "aid_2023", "aid_2026_proj", "tax_source", "new_tax", "health_share",
            "tax_for_health", "gov_spend", "ks_score", "ks_cluster", "room", "shift_cap", "shift_cap_source",
            "budget_shift", "capacity", "need", "need_source", "need_high", "need_tier", "lives_per_musd",
            "cost_per_life", "eff_index", "in_dashboard", "efficiency", "eff_source", "eff_high"]
    C = base[cols].copy()
    C["in_engine"] = C.index.isin(R["matched"])
    C = C.join(cpl[[c for c in cpl.columns if c.startswith(("s_", "l_"))]])
    for key, a in allocs.items():
        C = C.join(a.add_suffix(f"_{key}"))
    C.rename_axis("iso3").to_csv(OUT / "countries.csv")
    pd.DataFrame(summary).to_csv(OUT / "rules.csv", index=False)
    cpl_all.rename_axis("iso3").to_csv(OUT / "cost_per_life.csv")
    R["by_disease"].to_csv(OUT / "results_disease.csv", index=False)
    R["by_line"].to_csv(OUT / "results_services.csv", index=False)
    R["draws"].to_csv(OUT / "results_draws.csv", index=False)
    s = pd.DataFrame(summary).set_index("rule")
    D = R["by_disease"]
    tot = D.groupby(["rule", "scenario"])["deaths_5y"].sum()
    print(f"{len(R['matched'])} matched countries; lambda {s.at[PLAN, 'lambda']:.1%}, {s.at[PLAN, 'zero_aid']} at zero aid")
    print(tot.round(0).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
