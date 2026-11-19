"""
Health financing dashboard: navigation only. Run:  streamlit run app.py

The pages live in views/ (not pages/, which Streamlit would auto-load and bypass this navigation) (Dashboard, All Countries, Reallocation, Methods; views/3_Validation.py is kept but not in the navigation); theme.py holds the look.
"""
import streamlit as st

import health_model as hm
import theme as th

th.setup()

# Streamlit forgets a widget's value when the page that draws it is not shown. Writing these values back on every run
# keeps the model settings when switching pages (country and scenario are shared through sel_iso3 / sel_preset).
PERSIST_PREFIXES = ("m_", "mb_", "c1_", "c2_", "c3b_", "c4_year_", "al_")
NOT_PERSISTED = {"m_params", "m_custom_tbl", "mb_custom_tbl"}          # data editors cannot be set this way
for _k in list(st.session_state.keys()):
    if isinstance(_k, str) and _k.startswith(PERSIST_PREFIXES) and _k not in NOT_PERSISTED:
        try:
            st.session_state[_k] = st.session_state[_k]
        except st.errors.StreamlitAPIException:     # buttons, download buttons, uploaders: not settable, skip
            pass

# The aid data source (IHME or OECD) every page reads: the team's two projection workbooks, built into
# model_data/sources/<source>/ by announcement_model/workbook_inputs.py. Shown at the top of every page.
SRC_NOTES = {
    "ihme": "IHME Development Assistance for Health, 2023: aid from every donor that IHME traces to a country, "
            "including money that private NGOs and foundations raise themselves.",
    "oecd": "OECD CRS health aid (gross disbursements), 2023: what donor governments and multilaterals report to the "
            "OECD. It leaves out most private NGO money, so totals are smaller.",
}
if st.session_state.get("data_source") not in hm.SOURCES:
    st.session_state["data_source"] = hm.DEFAULT_SOURCE
st.session_state["data_source"] = st.session_state["data_source"]     # keep the choice across pages


pg = st.navigation([st.Page(path, title=title, icon=icon, default=(i == 0))
                    for i, (path, title, icon) in enumerate(th.PAGES)], position="top")
with st.container(key="src_bar"):
    _c1, _c2 = st.columns([1, 2.6], vertical_alignment="center")
    with _c1:
        # a radio styled as a two-button switch (theme.py, .st-key-src_bar): Streamlit's test harness can drive it
        st.radio("Aid Data Source", list(hm.SOURCES), format_func=hm.SOURCES.get, key="data_source", horizontal=True,
                 help="Switches every page between the two projection workbooks. Both apply the same announced cuts "
                      "and rules; only the 2023 aid data underneath differs.")
    with _c2:
        st.caption(SRC_NOTES[st.session_state["data_source"]])
pg.run()
