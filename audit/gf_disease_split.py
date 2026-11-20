"""
OECD data source fix (Nov 2026): the Global Fund's own split of its money between HIV, TB and malaria, by country, for
the OECD aid rows that come through the Global Fund. Read-only with respect to the model.

    python audit/gf_disease_split.py

Why: in the team's OECD projection workbook (OECD_aid_cut_projections.xlsx, sheet Base_Projection, columns AC-AE),
Global Fund money to 42 of 107 recipients has a TB share of exactly 0 (Ghana, Zambia, Malawi, South Africa, Ukraine,
Haiti, Sierra Leone, Senegal, ...). These are the countries where the Global Fund gives one combined TB/HIV grant (or a
multi-component grant) instead of separate ones: the OECD Creditor Reporting System records each grant under a single
purpose code, so the TB part shows up as HIV (or HIV and malaria). The Global Fund's own budgets split every grant by
the disease each budget line is for.

Source: Global Fund grant budgets (datasets/GlobalFund/GF_budgets_2026-10-14.csv, Global Fund data service; not
committed, 89 MB): budget lines for calendar year 2023 (BudgetYearFrom1 = 2023, every grant cycle), the baseline year
of both aid workbooks, summed by country (GeographyName1) and disease of the budget line (ModuleComponent1: HIV/AIDS,
Tuberculosis, Malaria; RSSH and Multi-Component lines are left out of the split). Shares = each disease / the three.

Writes audit/logic_review/gf_disease_split.csv (iso3, gf_hiv, gf_tb, gf_malaria shares summing to 1, the 2023 budget
for the three in US$, and whether the country had a combined TB/HIV or multi-component grant). Used by
announcement_model/workbook_inputs.py: each OECD Global Fund row keeps its HIV + TB + malaria total and divides it by
these shares (money doesn't move in or out of the three diseases, only between them).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GF = ROOT / "datasets" / "GlobalFund" / "GF_budgets_2026-10-14.csv"
WHO_TB = ROOT / "datasets" / "WHO_TB" / "TB_burden_countries_2026-10-14.csv"
OUT = ROOT / "audit" / "logic_review" / "gf_disease_split.csv"
YEAR = 2023
DIS = {"HIV/AIDS": "gf_hiv", "Tuberculosis": "gf_tb", "Malaria": "gf_malaria"}
# Global Fund names that differ from WHO's (Zanzibar has its own grants; the OECD records it as part of Tanzania)
NAME_FIX = {"Congo (Democratic Republic)": "COD", "Tanzania (United Republic)": "TZA", "Côte d'Ivoire": "CIV",
            "Bolivia (Plurinational State)": "BOL", "Iran (Islamic Republic)": "IRN", "Moldova": "MDA",
            "Lao People's Democratic Republic": "LAO", "Korea (Democratic Peoples Republic)": "PRK",
            "Venezuela": "VEN", "Venezuela (Bolivarian Republic)": "VEN", "Lao (Peoples Democratic Republic)": "LAO", "Kosovo": "KSV", "Zanzibar": "TZA", "Viet Nam": "VNM", "Congo": "COG",
            "Kyrgyzstan": "KGZ", "Palestine": "PSE", "Turkey": "TUR", "Türkiye": "TUR", "Syrian Arab Republic": "SYR",
            "Gambia": "GMB", "Sao Tome and Principe": "STP", "Eswatini": "SWZ", "Cabo Verde": "CPV",
            "Micronesia (Federated States)": "FSM", "Russian Federation": "RUS"}


def main():
    d = pd.read_csv(GF, usecols=["GeographyName1", "GrantComponent1", "BudgetYearFrom1", "ModuleComponent1",
                                 "BudgetAmount_ReferenceRate"], low_memory=False)
    d = d[d.BudgetYearFrom1 == YEAR]
    who = pd.read_csv(WHO_TB, usecols=["country", "iso3"]).drop_duplicates().set_index("country").iso3
    names = {n: NAME_FIX.get(n, who.get(n)) for n in d.GeographyName1.unique()}
    d["iso3"] = d.GeographyName1.map(names)
    unmatched = sorted(n for n, i in names.items() if i is None or pd.isna(i))
    d = d[d.iso3.notna()]
    t = d[d.ModuleComponent1.isin(DIS)].pivot_table(index="iso3", columns="ModuleComponent1",
                                                     values="BudgetAmount_ReferenceRate", aggfunc="sum")
    t = t.reindex(columns=list(DIS)).fillna(0.0).clip(lower=0)
    tot = t.sum(axis=1)
    out = t.div(tot, axis=0).rename(columns=DIS)
    out["gf_budget_3dis_usd"] = tot
    comb = d[d.GrantComponent1.isin(["TB/HIV", "Multicomponent"])].iso3.unique()
    out["combined_grant"] = out.index.isin(comb)
    out = out[tot > 0]
    out.index.name = "iso3"
    out.reset_index().round(6).to_csv(OUT, index=False)
    print(f"{len(out)} countries with 2023 Global Fund budgets for HIV, TB or malaria; "
          f"{int(out.combined_grant.sum())} with a combined TB/HIV or multi-component grant")
    print("not matched to an ISO3 code (regional grants etc.):", unmatched)
    print(out.loc[[i for i in ("GHA", "ZMB", "MWI", "ZAF", "UKR", "SLE", "SEN", "NGA", "IDN") if i in out.index]]
          .round(3).to_string())


if __name__ == "__main__":
    main()
