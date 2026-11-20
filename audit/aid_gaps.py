"""
Aid-data gaps check (Nov 2026): for every model country, both aid data sources (IHME, OECD) and each of HIV, TB and
malaria, flags holes and implausible values in the money the model cuts, and model results with no numbers.

    python audit/aid_gaps.py

Checks (one row per flag in audit/logic_review/aid_gaps.csv; a summary is printed):
  gf_hole      the Global Fund's 2023 budget gives the disease at least 5% of the country's HIV + TB + malaria money
               and at least US$0.5M (audit/logic_review/gf_disease_split.csv), but the model's Global Fund money for
               the disease is under 10% of that (the hole the OECD workbook had: TB 0 in 42 countries)
  gf_low_total the model's Global Fund money for the three diseases is under a third of the Global Fund's 2023 budget
               for them (money missing from the aid data, not just mis-split)
  tb_vs_who    the model's TB aid is under a fifth, or over five times, the external TB money the national programme
               reports to WHO (tb_prog_ext_share x tb_prog_funding_usd), where that is at least US$1M
  no_result    a disease with aid in the model whose full cut gives no deaths, or no people losing a service, or a
               missing (NaN) number in the results
Reads model_data/sources/<source>/dah_lines.csv and model_data/country_inputs.csv; runs health_model.run_country
(central values) for the result checks.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
import health_model as hm  # noqa: E402

OUT = ROOT / "audit" / "logic_review" / "aid_gaps.csv"
GF = ROOT / "audit" / "logic_review" / "gf_disease_split.csv"
DIS = {"HIV": "gf_hiv", "TB": "gf_tb", "Malaria": "gf_malaria"}
MAIN_LINE = {"HIV": "hiv_art", "TB": "tb_ds", "Malaria": "mal_itn", "Immunization": "imm"}


def money_flags(src: str, lines: pd.DataFrame, ci: pd.DataFrame, gf: pd.DataFrame) -> list:
    out = []
    g = lines[lines.chan_grp == "Global Fund"].groupby(["iso3", "bucket"]).usd.sum().unstack().fillna(0)
    for iso in sorted(set(lines.iso3) & set(gf.index)):
        have = g.loc[iso] if iso in g.index else pd.Series(0.0, index=list(DIS))
        tot_budget = float(gf.at[iso, "gf_budget_3dis_usd"])
        tot_model = float(sum(have.get(b, 0.0) for b in DIS))
        if tot_model < tot_budget / 3:
            out.append({"source": src, "iso3": iso, "check": "gf_low_total", "disease": "HIV+TB+malaria",
                        "model_usd": tot_model, "reference_usd": tot_budget,
                        "note": "Global Fund money in the aid data well below the Global Fund's 2023 budget"})
        for b, k in DIS.items():
            sh = float(gf.at[iso, k])
            ref = sh * tot_model                       # what the Global Fund split implies for the money recorded
            if sh >= 0.05 and sh * tot_budget >= 5e5 and have.get(b, 0.0) < 0.1 * ref:
                out.append({"source": src, "iso3": iso, "check": "gf_hole", "disease": b, "model_usd": have.get(b, 0.0),
                            "reference_usd": ref, "note": f"Global Fund budgets {sh:.0%} for {b}"})
    tb = lines[lines.bucket == "TB"].groupby("iso3").usd.sum()
    who = (ci.tb_prog_ext_share * ci.tb_prog_funding_usd)
    for iso in sorted(set(lines.iso3) & set(ci.index)):
        w, m = who.get(iso, np.nan), float(tb.get(iso, 0.0))
        if np.isfinite(w) and w >= 1e6 and (m < w / 5 or m > 5 * w):
            out.append({"source": src, "iso3": iso, "check": "tb_vs_who", "disease": "TB", "model_usd": m,
                        "reference_usd": w, "note": "TB aid vs external TB money reported to WHO"})
    return out


def result_flags(src: str, I: dict, P: pd.DataFrame) -> list:
    out = []
    L = I["lines"]
    for iso in sorted(set(L.iso3) & set(I["ci"].index)):
        for b in hm.BUCKETS:
            if L[(L.iso3 == iso) & (L.bucket == b)].usd.sum() <= 1e5:
                continue
            r = hm.run_country(iso, hm.uniform_scenario(b, 1.0), hm.Fiscal(), I, P, n_draws=0)
            bb = r["buckets"].set_index("bucket").loc[b]
            ln = r["lines"].set_index("line").loc[MAIN_LINE[b]]
            vals = {"deaths_5y": bb["deaths_5y"], "units_lost": ln["units_lost"], "net_loss_usd": bb["net_loss_usd"]}
            bad = [k for k, v in vals.items() if not np.isfinite(v)]
            if bad or vals["deaths_5y"] <= 0 or (vals["units_lost"] <= 0 and vals["net_loss_usd"] > 1e5):
                out.append({"source": src, "iso3": iso, "check": "no_result", "disease": b,
                            "model_usd": vals["net_loss_usd"], "reference_usd": np.nan,
                            "note": "; ".join(f"{k}={v:.3g}" for k, v in vals.items())})
    return out


def main():
    gf = pd.read_csv(GF).set_index("iso3")
    P = hm.load_params()
    rows = []
    for src in ("ihme", "oecd"):
        I = hm.load_inputs(source=src)
        rows += money_flags(src, I["lines"], I["ci"], gf) + result_flags(src, I, P)
    t = pd.DataFrame(rows, columns=["source", "iso3", "check", "disease", "model_usd", "reference_usd", "note"])
    t.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print(t.groupby(["source", "check"]).size().unstack(fill_value=0).to_string() if len(t) else "no flags")
    for c in ("gf_hole", "gf_low_total", "no_result"):
        x = t[t.check == c]
        if len(x):
            print(f"\n{c}:")
            print(x.assign(model_M=x.model_usd / 1e6, ref_M=x.reference_usd / 1e6)
                  [["source", "iso3", "disease", "model_M", "ref_M", "note"]].round(2).to_string(index=False))
    x = t[t.check == "tb_vs_who"]
    print("\ntb_vs_who:", ", ".join(f"{s}:{i}({m / 1e6:.1f} vs {w / 1e6:.1f})" for s, i, m, w in
                                     x[["source", "iso3", "model_usd", "reference_usd"]].itertuples(index=False)))


if __name__ == "__main__":
    main()
