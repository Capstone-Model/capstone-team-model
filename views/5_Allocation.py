"""
Reallocation: the team's allocation model. The same post-cut health aid pool, re-divided so countries that can raise
more money at home lose more aid, while countries where aid saves lives cheaply and high-burden countries are
protected; four scenarios run through this dashboard's engine.

Data: model_data/allocation/*.csv, written by allocation_model.py from "ALLOCATION MODEL/Allocation_model_CostPerLife_
Need.xlsx" (the method is in that folder's two PDFs). Nothing here runs the engine: rebuild with python allocation_model.py.
"""
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import allocation_model as am
import model_section as ms
import theme as th
from country_lists import ALLOWED_COUNTRIES
from model_section import chart, money, num, pill_row, rng

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "ALLOCATION MODEL"
e = html.escape

# scenario key -> (name, short name for chart axes, color). Order: the story (reallocation first, then new money)
SCEN = {"red": ("Current Split, No New Money", "Current Split,<br>No New Money", th.ROSE),
        "blue": ("Reallocated, No New Money", "Reallocated,<br>No New Money", th.BLUE),
        "yellow": ("Current Split + New Money", "Current Split<br>+ New Money", th.AMBER),
        "green": ("Reallocated + New Money", "Reallocated<br>+ New Money", th.TEAL)}
ORDER = list(SCEN)
RULE_PHRASE = {"cpl_need": "by ability to pay, cost per life saved and disease need",
               "cpl": "by ability to pay and cost per life saved",
               "need": "by ability to pay and disease need",
               "cap": "by ability to pay alone"}
SERVICE_NAMES = {"hiv_art": ("HIV Treatment (ART)", "people on ART"), "hiv_pmtct": ("PMTCT", "women"),
                 "tb_ds": ("TB Treatment", "courses"), "tb_dr": ("Drug-Resistant TB Treatment", "courses"),
                 "mal_itn": ("Bednets", "people"), "mal_irs": ("Indoor Spraying", "people"),
                 "mal_cm": ("Malaria Testing & Treatment", "cases treated"), "imm": ("Routine Immunization", "children")}
TIER_COLORS = {"High": th.MAP_RAMP[5], "Medium": th.MAP_RAMP[3], "Low": th.MAP_RAMP[2]}
SOURCE_NOTE = "Team allocation model (Allocation_model_CostPerLife_Need.xlsx); this dashboard's model for deaths."


@st.cache_data(show_spinner=False)
def _load():
    return am.load()


def dot(color: str) -> str:
    return f"<span class='dot' style='background:{color}'></span>"


def pct(x: float, d: int = 0) -> str:
    return f"{x:.{d}%}"


def m_usd(x_m: float) -> str:
    """US$ millions -> the dashboard's money format."""
    return money(x_m * 1e6)


R = _load()
C = R["countries"]
RULES = R["rules"]
N_ALL, N_MATCHED = len(C), int(C["in_engine"].sum())

# --------------------------------------------------------------------------- #
# Allocation rule (every number below follows it)
# --------------------------------------------------------------------------- #
rule_keys = list(am.RULES)
with st.container(key="alloc_controls", border=True):
    c1, _ = st.columns([1.6, 2])
    with c1:
        rule = st.selectbox("Allocation Rule", rule_keys, key="al_rule", format_func=lambda k: am.RULES[k][0],
                            help="How the cut is shared out. Every rule divides the same total cut; only who absorbs "
                                 "it changes. The team's plan uses all three factors.")
        st.markdown(f"<div style='font-size:13.5px;color:{th.MUTED};margin-top:-6px;line-height:1.45'>"
                    + ("Team plan · " if rule == am.PLAN else "") + f"Need strength k = {RULES.at[rule, 'need_k']:g} · efficiency strength k = "
                    f"{RULES.at[rule, 'eff_k']:g} · λ = {RULES.at[rule, 'lambda']:.1%}</div>",
                    unsafe_allow_html=True)

DD = am.deaths_by_disease(R, rule)
TOT = DD.loc["Total"]
RANGE = {s: am.deaths_range(R, rule, None, s) for s in ORDER}
SAVED = TOT["red"] - TOT["blue"]
SAVED_R = am.deaths_range(R, rule, "red", "blue")
cut_col, aid_col, eff_col = f"cut_{rule}", f"aid_2026_{rule}", f"eff_capacity_{rule}"
POOL = C["aid_2026_proj"].sum()
TOTAL_CUT = C["aid_2023"].sum() - POOL
CAPACITY = C["capacity"].sum()
ZERO = int(RULES.at[rule, "zero_aid"])
LAM = float(RULES.at[rule, "lambda"])
proj_cut = C["aid_2023"] - C["aid_2026_proj"]

# --------------------------------------------------------------------------- #
# Lede
# --------------------------------------------------------------------------- #
drop = SAVED / TOT["red"]
head = (f"Dividing the same {ms.money_words(POOL * 1e6)} of health aid {RULE_PHRASE[rule]} would cut extra deaths "
        f"by {drop:.0%}")
dek = (f"Health aid to these {N_ALL} countries is projected to fall by {ms.money_words(TOTAL_CUT * 1e6)} a year "
       f"between 2023 and 2026 (OECD). Keeping that smaller total fixed, this plan asks countries that can raise more "
       f"money at home to give up more aid"
       + {"cpl_need": ", and protects countries where aid saves lives cheaply and countries with a high disease burden.",
          "cpl": ", and protects countries where aid saves lives cheaply.",
          "need": ", and protects countries with a high disease burden.",
          "cap": "."}[rule]
       + f" With no new money, extra deaths over 2026-2030 in the {N_MATCHED} countries this dashboard models fall "
         f"from {ms.round_words(TOT['red'])} to {ms.round_words(TOT['blue'])}.")
big = [(num(TOT["red"]), "Extra Deaths, Current Split", "2026-2030, no new money", ""),
       (num(TOT["blue"]), "Extra Deaths, Reallocated", "same aid, no new money", ""),
       (num(SAVED), "Lives Saved by Reallocation Alone", f"range {rng(*SAVED_R)}", ""),
       (str(ZERO), "Countries Moved to Zero Aid", f"of {N_ALL}; covered at home if new money is raised", "")]
st.markdown(
    ("<div class='ed-lede'><div class='ed-kicker'>Allocation Model · Who Should Absorb the Aid Cut?</div>"
     f"<div class='ed-headline'>{e(head)}</div><p class='ed-dek'>{e(dek)}</p></div>"
     "<div class='ed-bignums'>"
     + "".join(f"<div class='ed-bignum'><div class='v {c}'>{e(v)}</div><div class='l'>{e(l)}</div>"
               f"<div class='n'>{e(n)}</div></div>" for v, l, n, c in big)
     + "</div>").replace("$", "&#36;"), unsafe_allow_html=True)
st.markdown(th.source_html(SOURCE_NOTE, f"Deaths: central estimate; ranges are the {th.RANGE_LABEL} "
                                        "(2.5th-97.5th percentile of 400 draws)."), unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# 1. The four scenarios
# --------------------------------------------------------------------------- #
th.section_header("grid_view", "The Four Scenarios",
                  help="Two choices, crossed: keep the current aid split or reallocate it, and add new domestic "
                       "money or not. Every scenario has the same total donor aid.")
fills = {"red": 0.0, "blue": 0.0, "yellow": np.minimum(C["capacity"], proj_cut).sum(),
         "green": np.minimum(C["capacity"], C[cut_col]).sum()}
desc = {"red": "Countries get the OECD-projected 2026 aid and raise nothing new. The whole cut is a hole in their "
               "health budgets.",
        "blue": "The same aid, divided by the allocation rule. Countries raise nothing new.",
        "yellow": "The OECD-projected aid, and each country uses its new domestic money for health to fill as much of "
                  "the hole as it can.",
        "green": "Aid divided by the allocation rule, and new domestic money fills the hole."}
cards = []
for s in ORDER:
    name, _, col = SCEN[s]
    lo, hi = RANGE[s]
    foot = (f"Aid {m_usd(POOL)} a year" + (f" · new money fills {m_usd(fills[s])}" if fills[s] else " · no new money"))
    t1, t2 = name.replace(" + ", ", + ").split(", ", 1)
    cards.append(f"<div class='ed-card' style='border-top-color:{col}'><div class='t'>{e(t1)}</div>"
                 f"<div class='s1' style='color:{col}'>{e(t2.capitalize())}</div>"
                 f"<div class='s'>{e(desc[s])}</div><div class='big'>{num(TOT[s])}</div>"
                 f"<div class='lbl'>extra deaths 2026-2030"
                 + (f" · range {rng(lo, hi)}" if TOT[s] >= 0.5 else "") + "</div>"
                 f"<div class='foot'>{e(foot)}</div></div>")
st.markdown(("<div class='ed-cards'>" + "".join(cards) + "</div>").replace("$", "&#36;"), unsafe_allow_html=True)

st.markdown("")
fig = go.Figure(go.Bar(
    x=[SCEN[s][1] for s in ORDER], y=[TOT[s] for s in ORDER], marker_color=[SCEN[s][2] for s in ORDER],
    marker_line_width=0, width=0.55,
    error_y=dict(type="data", symmetric=False, color=th.MUTED, thickness=1.3, width=7,
                 array=[max(RANGE[s][1] - TOT[s], 0) for s in ORDER],
                 arrayminus=[max(TOT[s] - RANGE[s][0], 0) for s in ORDER]),
    customdata=[[SCEN[s][0], rng(*RANGE[s])] for s in ORDER],
    hovertemplate="<b>%{customdata[0]}</b><br>%{y:,.0f} extra deaths<br>range %{customdata[1]}<extra></extra>"))
for s in ORDER:
    top = max(RANGE[s][1], TOT[s])
    label = f"<b>{num(TOT[s])}</b>"
    if s != "red":
        label = (f"<span style='color:{th.MUTED}'><b>−{1 - TOT[s] / TOT['red']:.0%}</b></span><br>" + label)
    fig.add_annotation(x=SCEN[s][1], y=top, text=label, showarrow=False, yshift=22 if s != "red" else 12,
                       font=dict(size=13, color=th.INK))
fig.update_layout(height=450, showlegend=False, margin=dict(t=105, b=10),
                  title=th.title_sub("Extra Deaths From Aid Cuts, 2026-2030",
                                     f"{N_MATCHED} countries · % change from the current split,<br>no new money"),
                  yaxis=dict(title="", tickformat=",.0f", range=[0, max(r[1] for r in RANGE.values()) * 1.22]))
chart(fig, source=SOURCE_NOTE,
      note=f"Whiskers: the {th.RANGE_LABEL}. Relative to 2021-2023 aid; includes HIV deaths from new infections.")

fig = go.Figure()
for b in am.BUCKETS:
    fig.add_bar(name=b, x=[SCEN[s][1] for s in ORDER], y=[DD.at[b, s] for s in ORDER],
                marker_color=th.DISEASE_COLORS[b], width=0.55,
                hovertemplate=f"<b>{b}</b><br>%{{y:,.0f}} extra deaths<extra></extra>")
for s in ORDER:
    fig.add_annotation(x=SCEN[s][1], y=TOT[s], text=f"<b>{num(TOT[s])}</b>", showarrow=False, yshift=12,
                       font=dict(size=13, color=th.INK))
fig.update_layout(barmode="stack", height=430, showlegend=False,
                  title=th.title_sub("Extra Deaths by Disease, 2026-2030",
                                     th.color_key((b, th.DISEASE_COLORS[b]) for b in am.BUCKETS)),
                  yaxis=dict(tickformat=",.0f", range=[0, TOT.max() * 1.12]))
chart(fig, source=SOURCE_NOTE)
tb_r, tb_b = DD.at["TB", "red"], DD.at["TB", "blue"]
if tb_b > tb_r:
    st.markdown(ms._esc(
        f"Reallocation alone saves lives from HIV, malaria and immunization, but TB deaths rise from "
        f"{tb_r:,.0f} to {tb_b:,.0f}: the reallocated aid comes partly from large middle-income countries with heavy "
        f"TB burdens, where this model's TB treatment saves lives cheaply. New domestic money closes those gaps in "
        f"the last scenario."))

# two-order decomposition of the combined change
nm = ((TOT["red"] - TOT["yellow"]) + (TOT["blue"] - TOT["green"])) / 2
ra = ((TOT["red"] - TOT["blue"]) + (TOT["yellow"] - TOT["green"])) / 2
dec_tip = ("New money and reallocation interact, so each one's effect depends on which comes first. These are the "
           "averages over both orders: new money = [(current split: no new money − with new money) + (reallocated: no "
           "new money − with new money)] / 2; reallocation likewise. An accounting split of these scenarios, not a "
           "causal estimate.")
pill_row([("New Money's Share of the Lives Saved", num(nm), "average of both orders", dec_tip),
          ("Reallocation's Share of the Lives Saved", num(ra), "average of both orders", dec_tip),
          ("Both Together", num(TOT["red"] - TOT["green"]), "current split → reallocated + new money", dec_tip)])

# --------------------------------------------------------------------------- #
# 2. How the cut is divided
# --------------------------------------------------------------------------- #
th.section_header("functions", "How the Cut Is Divided",
                  help="The total cut stays fixed; the model only changes how it is split between countries.")
nk, ek = RULES.at[rule, "need_k"], RULES.at[rule, "eff_k"]
need_t = rf"\times \dfrac{{1}}{{1+{nk:g}\,\text{{Need}}_i}}" if nk else ""
eff_t = rf"\times \dfrac{{1}}{{1+{ek:g}\,\text{{Efficiency}}_i}}" if ek else ""
st.latex(rf"\begin{{aligned}}\text{{Cut}}_i &= \min\left(\text{{Aid}}^{{2023}}_i,\;\lambda \times \text{{Effective}}_i\right)"
         rf"\\[6pt] \text{{Effective}}_i &= \text{{Capacity}}_i {need_t} {eff_t}\end{{aligned}}")
best = C.loc[C["in_dashboard"]].sort_values("cost_per_life").iloc[0]
steps = [
    ("1 · Capacity", m_usd(CAPACITY), "a year, all countries",
     "New tax revenue (Benitez et al. ceiling, Akitoby et al. pace, 2027-30 average) times the country's current "
     "health share of government spending, plus a budget shift toward health: up to the country's best 4-year rise "
     "in health's share, phased in over 2027-2030 and scaled by its Kharas & Sarkar fiscal space score (none at or "
     "below 48.8, full at 63.9 and above).", th.BRAND),
    ("2 · Need Discount", f"1 / (1 + {nk:g} × need)" if nk else "Off",
     "high-burden countries asked for less",
     "Need: the team disease-burden model's DAH-adjusted need index (80% HIV/TB/malaria burden, 20% aid exposure). "
     f"It runs from 0 to {C['need'].max():.2f}, so the neediest country's capacity counts at "
     f"{1 / (1 + (nk or RULES.at[am.PLAN, 'need_k']) * C['need'].max()):.0%}"
     + ("." if nk else f" when the discount is on (k = {RULES.at[am.PLAN, 'need_k']:g}).")
     + " Countries outside that model take the median of their capstone need tier.", th.BRAND),
    ("3 · Efficiency Discount", f"1 / (1 + {ek:g} × efficiency)" if ek else "Off",
     "where aid saves lives cheaply",
     f"Lives saved per US$1M of aid lost in this dashboard (HIV, TB, malaria and immunization, weighted by each "
     f"country's aid mix), divided by the highest ({best['country']}, {money(best['cost_per_life'])} per life saved), "
     f"so efficiency runs 0-1. The {int((~C['in_dashboard']).sum())} countries not in the dashboard take the average "
     f"of their income group.", th.BRAND),
    ("4 · λ", f"{LAM:.1%}", f"of effective capacity; {ZERO} countries to zero",
     "λ is one number for every country: the share of its effective capacity it gives up. It is raised until the "
     f"cuts add up to exactly {m_usd(TOTAL_CUT)}. A country whose cut would exceed all its aid is cut to zero "
     "instead, and λ rises a little so the others cover the difference.", th.BRAND)]
st.markdown(("<div class='ed-cards'>" + "".join(
    f"<div class='ed-card' style='border-top-color:{col}'><div class='t'>{e(t)}</div>"
    f"<div class='big' style='font-size:21px'>{e(v)}</div><div class='lbl'>{e(n)}</div>"
    f"<div class='foot'>{e(tip)}</div></div>" for t, v, n, tip, col in steps) + "</div>").replace("$", "&#36;"),
    unsafe_allow_html=True)

# where the aid moves (reallocated + new money): each country's 2023 aid split four ways
st.markdown("")
A = C.copy()
A["model"], A["proj"] = A[aid_col], A["aid_2026_proj"]
A["keep"] = np.minimum(A["model"], A["proj"])
A["gain"] = (A["model"] - A["proj"]).clip(lower=0)
A["away"] = (A["proj"] - A["model"]).clip(lower=0)
A["rest"] = A["aid_2023"] - np.maximum(A["model"], A["proj"])
A["rank"] = A[eff_col] / A["aid_2023"]
A = A.sort_values("rank")
parts = [("keep", "Aid the country keeps", th.blend(th.BLUE, 0.38)),
         ("gain", "Aid gained in the reallocation", th.BLUE),
         ("away", "Aid moved away, replaced by new money", th.TEAL),
         ("rest", "Rest of the cut, replaced by new money", th.blend(th.TEAL, 0.4))]
fig = go.Figure()
cd = np.stack([A["country"], A["aid_2023"].map(m_usd), A["proj"].map(m_usd), A["model"].map(m_usd)], axis=-1)
for k, lbl, col in parts:
    fig.add_bar(name=lbl, x=A["country"], y=A[k] / A["aid_2023"] * 100, marker_color=col, marker_line_width=0,
                customdata=np.column_stack([cd, (A[k]).map(m_usd)]),
                hovertemplate=("<b>%{customdata[0]}</b><br>" + lbl + ": %{y:.1f}% (%{customdata[4]})<br>"
                               "2023 aid %{customdata[1]} · OECD 2026 %{customdata[2]} · this plan %{customdata[3]}"
                               "<extra></extra>"))
mv = (A["model"] - A["proj"])
flag = list(mv.nlargest(2).index) + list(mv.nsmallest(2).index)
pos = {iso: list(A.index).index(iso) for iso in flag}
for k, iso in enumerate(sorted(flag, key=pos.get)):            # neighbours alternate heights so labels never overlap
    fig.add_annotation(x=A.at[iso, "country"], y=100, yshift=4, text=A.at[iso, "country"], showarrow=True,
                       arrowhead=0, arrowcolor=th.MUTED, ax=0, ay=-18 if k % 2 == 0 else -34,
                       font=dict(size=11, color=th.MUTED))
n_gain, n_lose = int(RULES.at[rule, "gain_vs_proj"]), int(RULES.at[rule, "lose_vs_proj"])
fig.update_layout(barmode="stack", bargap=0.12, height=500, margin=dict(t=135),
                  title=th.title_sub("Where the Aid Moves, With New Money",
                                     f"Each bar is one country's 2023 health aid<br>{n_gain} gain aid and {n_lose} "
                                     f"lose aid vs the OECD projection"),
                  xaxis=dict(showticklabels=False, title="← Least able to replace aid · most able →"),
                  yaxis=dict(range=[0, 112], ticksuffix="%", title="Share of 2023 health aid"))
chart(fig, source=SOURCE_NOTE,
      note="Ranked by effective capacity relative to 2023 aid. Hover for each country. Every country's cut is "
           "smaller than its capacity, so new money covers all of it in this scenario.")

# --------------------------------------------------------------------------- #
# 3. Who is left with a funding gap
# --------------------------------------------------------------------------- #
th.section_header("money_off", "Who Is Left With a Funding Gap",
                  help="Funding gap: aid lost since 2023 that new domestic money does not replace, US$ a year, all "
                       f"{N_ALL} countries.")
gap = {"red": proj_cut, "blue": C[cut_col], "yellow": (proj_cut - C["capacity"]).clip(lower=0),
       "green": (C[cut_col] - C["capacity"]).clip(lower=0)}
tiers = C["need_tier"]
fig = go.Figure()
for t in ["High", "Medium", "Low"]:
    n_t = int((tiers == t).sum())
    fig.add_bar(name=f"{t} disease need ({n_t} countries)", x=[SCEN[s][1] for s in ORDER],
                y=[gap[s][tiers == t].sum() / 1e3 for s in ORDER], marker_color=TIER_COLORS[t], width=0.55,
                hovertemplate=f"<b>{t} need</b><br>%{{y:$,.2f}}B a year<extra></extra>")
for s in ORDER:
    fig.add_annotation(x=SCEN[s][1], y=gap[s].sum() / 1e3, text=f"<b>{m_usd(gap[s].sum())}</b>", showarrow=False,
                       yshift=12, font=dict(size=13, color=th.INK))
fig.update_layout(barmode="stack", height=470, showlegend=False, margin=dict(t=110),
                  title=th.title_sub("Funding Gap Left Each Year, by Disease Need",
                                     "US$ a year, all 82 countries<br>" + th.color_key(
                                         (f"{t} need ({int((tiers == t).sum())})", TIER_COLORS[t])
                                         for t in ["High", "Medium", "Low"])),
                  yaxis=dict(tickformat="$,.0f", ticksuffix="B", range=[0, TOTAL_CUT / 1e3 * 1.15]))
chart(fig, source=SOURCE_NOTE,
      note=f"Disease need: thirds of the {N_ALL} countries by the need index; countries tied at a fallback value "
           "share a third, so the groups are not equal.")
hi_r, hi_b = gap["red"][tiers == "High"].sum(), gap["blue"][tiers == "High"].sum()
lo_r, lo_b = gap["red"][tiers == "Low"].sum(), gap["blue"][tiers == "Low"].sum()
st.markdown(ms._esc(f"With no new money, reallocation shrinks the high-need countries' gap from {m_usd(hi_r)} to "
                    f"{m_usd(hi_b)} a year and grows the low-need countries' gap from {m_usd(lo_r)} to "
                    f"{m_usd(lo_b)}. The total gap is the same {m_usd(TOTAL_CUT)}: only who carries it changes."))

# --------------------------------------------------------------------------- #
# 4. Country by country
# --------------------------------------------------------------------------- #
th.section_header("public", "Country by Country",
                  help="Every number for one country, then the full table. Aid figures are all health aid (OECD), US$ "
                       "a year; deaths cover HIV, TB, malaria and immunization in the countries this dashboard models.")
CR = am.country_results(R, rule)
names = C["country"].sort_values()
cur = st.session_state.get("sel_iso3", "NGA")
default = cur if cur in C.index else "NGA"
if st.session_state.get("al_country") not in C.index:
    st.session_state["al_country"] = default
c1, _ = st.columns([1.3, 2])
with c1:
    iso = st.selectbox("Country", list(names.index), key="al_country", format_func=lambda i: C.at[i, "country"])
if iso in ALLOWED_COUNTRIES:
    st.session_state["sel_iso3"] = iso                 # the other pages follow (only countries they cover)
r = C.loc[iso]
cut = r[cut_col]
ncut, cutp = r["aid_2023"] - r["aid_2026_proj"], cut / r["aid_2023"] if r["aid_2023"] else 0.0
nd, ed = r[f"need_disc_{rule}"], r[f"eff_disc_{rule}"]
cpl_txt = (f"{money(r['cost_per_life'])} per life saved" if r["in_dashboard"]
           else f"not in the dashboard: {r['income'].lower()} average")
pill_row([("2023 Aid", m_usd(r["aid_2023"]), "a year, all health aid"),
          ("OECD 2026 Projection", m_usd(r["aid_2026_proj"]), f"cut {pct(ncut / r['aid_2023'], 1)}"),
          ("This Plan, 2026", m_usd(r[aid_col]), f"cut {pct(cutp, 1)}"),
          ("Change vs Projection", ("+" if r[aid_col] >= r["aid_2026_proj"] else "") + m_usd(r[aid_col] - r["aid_2026_proj"]),
           "a year")], 4)
dshare = ("no capacity counted" if r["capacity"] <= 0 else
          "all new tax revenue" if r["budget_shift"] <= 0 else
          f"{r['budget_shift'] / r['capacity']:.0%} from the budget shift")
pill_row([("Capacity", m_usd(r["capacity"]), dshare,
           f"New tax revenue {m_usd(r['new_tax'])} × health share "
           f"{r['health_share']:.1%} = {m_usd(r['tax_for_health'])}; budget shift {m_usd(r['budget_shift'])} "
           f"(fiscal space room {r['room']:.0%} × best 4-year rise {r['shift_cap']:.2%} ({r['shift_cap_source']}) × "
           f"government spending × 0.625 phase-in)." if r["capacity"] > 0 else
           "No new tax revenue or budget shift is counted (missing data or already at tax potential)."),
          ("Disease Need", f"{r['need']:.2f}", f"discount × {nd:.2f}", r["need_source"]),
          ("Efficiency", f"{r['efficiency']:.2f}", f"discount × {ed:.2f}", cpl_txt),
          ("Effective Capacity", m_usd(r[eff_col]),
           f"× λ {LAM:.1%} → cut {m_usd(cut)}" if cut < r["aid_2023"] - 1e-6 else "cut capped at all 2023 aid",
           "Capacity × need discount × efficiency discount. The cut is λ times this, but never more than all of the "
           "country's 2023 aid.")], 4)
if iso in CR.index:
    cr = CR.loc[iso]
    pill_row([(f"Extra Deaths: {SCEN[s][0]}", num(cr[f'deaths_{s}']), "2026-2030") for s in ORDER], 4)
else:
    st.caption(f"{r['country']} is not one of the countries this dashboard models, so no deaths are estimated.")
st.markdown(ms._esc(
    f"**How {r['country']}'s cut is set.** Capacity {m_usd(r['capacity'])} a year, times the need discount "
    f"{nd:.2f} and the efficiency discount {ed:.2f}, gives an effective capacity of {m_usd(r[eff_col])}. "
    + (f"Its cut is λ = {LAM:.1%} of that, {m_usd(cut)}, or {pct(cutp, 1)} of its {m_usd(r['aid_2023'])} of 2023 aid."
       if cut < r["aid_2023"] - 1e-6 else
       f"λ = {LAM:.1%} of that would exceed all of its {m_usd(r['aid_2023'])} of 2023 aid, so its aid goes to zero "
       f"and its own money is expected to replace it.")))

T = pd.DataFrame({
    "Country": C["country"], "Income Group": C["income"],
    "2023 Aid ($M)": C["aid_2023"], "OECD 2026 ($M)": C["aid_2026_proj"], "This Plan 2026 ($M)": C[aid_col],
    "Change vs OECD ($M)": C[aid_col] - C["aid_2026_proj"], "Capacity ($M/yr)": C["capacity"],
    "Need": C["need"], "Cost per Life Saved ($)": C["cost_per_life"], "Effective Capacity ($M)": C[eff_col],
    "Deaths, Current Split": CR["deaths_red"].reindex(C.index),
    "Deaths, Reallocated": CR["deaths_blue"].reindex(C.index)})
T["Lives Saved"] = T["Deaths, Current Split"] - T["Deaths, Reallocated"]
T = T.sort_values("Change vs OECD ($M)", ascending=False).round(
    {c: 1 for c in T.columns if "($M" in c} | {"Need": 2, "Cost per Life Saved ($)": 0, "Deaths, Current Split": 0,
                                                 "Deaths, Reallocated": 0, "Lives Saved": 0})
fm = st.column_config.NumberColumn
st.dataframe(T, hide_index=True, height=420, **ms.WIDE, column_config={
    "Country": st.column_config.TextColumn(pinned=True, width="medium"),
    "2023 Aid ($M)": fm(format="localized"), "OECD 2026 ($M)": fm(format="localized"),
    "This Plan 2026 ($M)": fm(format="localized"), "Change vs OECD ($M)": fm(format="%+.1f"),
    "Capacity ($M/yr)": fm(format="localized"), "Need": fm(format="%.2f"),
    "Cost per Life Saved ($)": fm(format="localized", help="Blank: not in the dashboard (uses its income-group "
                                                            "average efficiency)"),
    "Effective Capacity ($M)": fm(format="localized"), "Deaths, Current Split": fm(format="localized"),
    "Deaths, Reallocated": fm(format="localized"), "Lives Saved": fm(format="localized")})
st.markdown(th.source_html(SOURCE_NOTE, "Deaths: no new money, 2026-2030. Blank deaths: the country is not modelled "
                                        "by this dashboard. Sorted by change versus the OECD projection."),
            unsafe_allow_html=True)
st.download_button("Download This Table (CSV)", T.to_csv(index=False).encode(), file_name=f"allocation_{rule}.csv",
                   mime="text/csv", key="al_dl_table")

# --------------------------------------------------------------------------- #
# 5. Services and coverage
# --------------------------------------------------------------------------- #
th.section_header("medical_services", "Services and Coverage",
                  help=f"Coverage across the {N_MATCHED} modelled countries, weighted by each country's population in "
                       "need of the service. Reallocated + new money keeps every service at its 2021-2023 coverage.")
COV, UN = am.coverage(R, rule), am.units_lost(R, rule)
head_cells = "".join(f"<th>{dot(SCEN[s][2])}{e(SCEN[s][0])}</th>" for s in ORDER)
body = []
for ln in am.SERVICES:
    nm_, unit = SERVICE_NAMES[ln]
    cells = []
    for s in ORDER:
        lost = UN.at[ln, s]
        cells.append(f"<td><span class='big'>{COV.at[ln, s]:.1%}</span><br>"
                     f"<span class='nm'>{num(lost) + ' lost' if lost >= 0.5 else 'none lost'}</span></td>")
    body.append(f"<tr><td class='svc'><b>{e(nm_)}</b><br><span class='sub'>units: {e(unit)} a year</span></td>"
                + "".join(cells) + "</tr>")
tot_cells = "".join(f"<td><b>{num(UN[s].sum())}</b></td>" for s in ORDER)
body.append(f"<tr class='tot'><td class='svc'><b>Service Units Lost a Year</b><br><span class='sub'>not unique "
            f"people: units differ and overlap</span></td>{tot_cells}</tr>")
st.markdown(("<div class='ed-nyt-wrap'><table class='ed-nyt' style='min-width:760px'><colgroup><col style='width:28%'>"
             + "<col style='width:18%'>" * 4 + f"</colgroup><thead><tr><th>Service</th>{head_cells}</tr></thead><tbody>"
             + "".join(body) + "</tbody></table></div>").replace("$", "&#36;"), unsafe_allow_html=True)
st.markdown(th.source_html(SOURCE_NOTE, "Coverage now minus the modelled drop, average 2026-2030; units lost a year "
                                        "relative to 2021-2023 coverage. The model counts losses only: money above "
                                        "2023 aid buys no extra coverage here."), unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# 6. Comparing allocation rules
# --------------------------------------------------------------------------- #
th.section_header("compare_arrows", "Comparing Allocation Rules",
                  help="Same aid pool, same new money, same model: only the rule that divides the cut changes. With "
                       "new money every rule closes every gap, so the rules differ only without it.")
rows = []
for k in rule_keys:
    d = am.deaths_by_disease(R, k).loc["Total"]
    rows.append({"key": k, "label": am.RULES[k][0], "deaths": d["blue"], "saved": d["red"] - d["blue"],
                 "range": am.deaths_range(R, k, "red", "blue"), "zero": int(RULES.at[k, "zero_aid"]),
                 "lam": float(RULES.at[k, "lambda"]),
                 "vs_cap": (am.deaths_by_disease(R, "cap").at["Total", "blue"] - d["blue"]),
                 "vs_cap_r": am.rule_gain_range(R, "cap", k) if k != "cap" else None})
RR = pd.DataFrame(rows).iloc[::-1]
fig = go.Figure(go.Bar(
    y=[f"{l}<br><b>{num(v)}</b>" for l, v in zip(RR["label"], RR["saved"])], x=RR["saved"], orientation="h",
    width=0.6, marker_line_width=0,
    marker_color=[th.BLUE if k == rule else th.blend(th.BLUE, 0.35) for k in RR["key"]],
    error_x=dict(type="data", symmetric=False, color=th.MUTED, thickness=1.3, width=6,
                 array=[r_[1] - s_ for r_, s_ in zip(RR["range"], RR["saved"])],
                 arrayminus=[s_ - r_[0] for r_, s_ in zip(RR["range"], RR["saved"])]),
    customdata=np.column_stack([RR["range"].map(lambda t: rng(*t)), RR["zero"], RR["lam"].map(lambda x: f"{x:.1%}")]),
    hovertemplate=("%{x:,.0f} lives saved vs the current split<br>range %{customdata[0]}<br>"
                   "λ %{customdata[2]} · %{customdata[1]} countries at zero aid<extra></extra>")))
fig.update_layout(height=360, showlegend=False, margin=dict(t=80, r=20),
                  title=th.title_sub("Lives Saved by Reallocation Alone, by Rule",
                                     "2026-2030, compared with the current split, no new money"),
                  xaxis=dict(tickformat="~s", nticks=5, range=[0, max(r_[1] for r_ in RR["range"]) * 1.08]),
                  yaxis=dict(ticklabelstandoff=10))
chart(fig, source=SOURCE_NOTE, note=f"Whiskers: the {th.RANGE_LABEL}.")
tbl = "".join(
    f"<tr{' style=\"background:' + th.SURFACE_TINT + '\"' if q['key'] == rule else ''}><td class='svc'><b>{e(q['label'])}"
    f"</b>{' <span class=\"sub\">(team plan)</span>' if q['key'] == am.PLAN else ''}</td>"
    f"<td>{q['lam']:.1%}</td><td>{q['zero']}</td><td>{num(q['deaths'])}</td>"
    f"<td>{'—' if q['vs_cap_r'] is None else '+' + num(q['vs_cap']) + '<br><span class=nm>' + rng(*q['vs_cap_r']) + '</span>'}</td></tr>"
    for _, q in RR.iloc[::-1].iterrows())
st.markdown(("<div class='ed-nyt-wrap'><table class='ed-nyt' style='min-width:700px'><colgroup><col style='width:36%'>"
             + "<col style='width:16%'>" * 4 + "</colgroup><thead><tr><th>Rule</th><th>λ</th><th>Countries at Zero Aid"
             "</th><th>Extra Deaths, No New Money</th><th>Lives Saved vs Ability to Pay Only</th></tr></thead><tbody>"
             + tbl + "</tbody></table></div>").replace("$", "&#36;"), unsafe_allow_html=True)

MED = am.disease_medians(R)
st.markdown("<div class='ed-kicker' style='margin:1.6rem 0 0.4rem'>What drives the efficiency ranking</div>",
            unsafe_allow_html=True)
pill_row([(b, f"${MED.at[b, 'cost_per_life']:,.0f}", "per life saved",
           f"{MED.at[b, 'lives_per_musd']:,.0f} lives saved per US$1M of aid lost. Median over the {int(MED.at[b, 'countries'])} of the dashboard's 70 countries where {b} is at least 5% of their HIV, "
           "TB, malaria and immunization aid. A country's efficiency weights these by its own aid mix.")
          for b in am.BUCKETS], 4)
st.markdown(ms._esc("TB treatment is this model's cheapest way to save a life, so countries with large TB programs "
                    "score as efficient and are protected. That is why the rules that use cost per life save the most "
                    "lives in this comparison, and why they lower TB coverage less."))

# --------------------------------------------------------------------------- #
# 7. Reading these results
# --------------------------------------------------------------------------- #
th.section_header("report", "Reading These Results")
# the team's results PDF (11 November 2026): extra deaths 2026-2030, current split / + new money / team plan reallocated
PDF_DEATHS = {"red": 898_897, "yellow": 199_893, "blue": 506_894}
_now = am.deaths_by_disease(R, am.PLAN).loc["Total"]
_gap = max(abs(_now[k] / v - 1) for k, v in PDF_DEATHS.items())
_same = all(round(_now[k]) == v for k, v in PDF_DEATHS.items())       # the PDF prints whole deaths
PDF_NOTE = ("Results use this dashboard's current model and match the team's results PDF exactly." if _same else
            f"Results use this dashboard's current model, which has been updated since the team's results PDF was "
            f"made: total deaths differ from the PDF by up to {_gap:.1%}, and some service counts (such as indoor "
            f"spraying) by more.")
st.markdown(ms._esc(f"""
- **Built to score well here.** The efficiency factor comes from the same model that scores the results, so the
  plan is designed to look good by this measure. Treat its lives saved as an upper bound; the need-only rule is the
  more independent comparison.
- **Losses only.** The model estimates what is lost when aid falls below its 2021-2023 level. Money above that level
  is not credited with extra coverage or lives, so the {m_usd(CAPACITY - fills['green'])} a year of new money beyond
  what fills the cut adds nothing to these results.
- **Two country sets.** The allocation covers {N_ALL} countries and all health aid (OECD); deaths and coverage cover the
  {N_MATCHED} of them this dashboard models, and only HIV, TB, malaria and immunization.
- **Capacity is a judgement.** New tax revenue and the budget shift are what each country could raise, not what it
  will. The Kharas & Sarkar index measures room for new spending, not budget flexibility; using it makes the budget
  shift smaller and the plan more conservative. Kosovo has no health-share data, so it gets no capacity and is never
  cut; Azerbaijan and Somalia have no fiscal space score, so they get no budget shift (their tax capacity still
  counts).
- **The strength of the discounts (k = 4) is a choice.** At k = 4 an efficiency score of 1 counts capacity at a fifth;
  need tops out at {C['need'].max():.2f}, so the neediest country counts at about a third.
- **Fixed scenarios, not forecasts.** {PDF_NOTE}
"""))
st.markdown("<div class='ed-kicker' style='margin:1.4rem 0 0.4rem'>Source files</div>", unsafe_allow_html=True)
cols = st.columns(3)
for col, (fname, label, mime) in zip(cols, [
        ("Allocation_model_CostPerLife_Need.xlsx", "Workbook (Excel)",
         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("Allocation_Model_Explanation.pdf", "Method (PDF)", "application/pdf"),
        ("Allocation Results.pdf", "Results (PDF)", "application/pdf")]):
    p = SRC / fname
    if p.exists():
        with col:
            st.download_button(label, p.read_bytes(), file_name=fname, mime=mime, key=f"al_dl_{fname}", **ms.WIDE)
