# streamlit_app.py — OpenFlow Data + Docs (no server-side filtering)

import json
import datetime as dt
import pandas as pd
import streamlit as st

from snowflake.snowpark.context import get_active_session
session = get_active_session()

# ---------------------------- CONFIG ----------------------------
DB = "ANALYTICS_FA"
SCHEMA = "RAW"
SEARCH_SERVICE = "ANALYTICS_FA.RAW.DOCS_SEARCH_FA"   # Cortex Search service name
DOCS_STAGE = "RAW_DOCS_STAGE"                        # stage for file preview
PREVIEW_SECONDS = 3600                               # presigned URL validity
PREVIEW_LIMIT = 100                                  # row preview cap

st.set_page_config(page_title="OpenFlow Data + Docs", layout="wide")
st.title("OpenFlow Data + Docs")

# ---------------------------- BASIC HELPERS ---------------------
def sql_to_df(sql: str, params=None) -> pd.DataFrame:
    return session.sql(sql, params=params).to_pandas()

def sql_scalar(sql: str, params=None):
    """Return the first column of the first row (or None)."""
    rows = session.sql(sql, params=params).collect()
    if not rows:
        return None
    row = rows[0].as_dict()
    return next(iter(row.values()))

def parse_variant_json(v):
    """Handle VARIANT or JSON string values into Python objects."""
    if v is None:
        return {}
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {"raw": v}
    try:
        json.dumps(v)
        return v
    except Exception:
        return {"raw": str(v)}

def get_presigned_url(path: str, seconds: int = PREVIEW_SECONDS) -> str | None:
    """
    Generate a presigned URL for a file stored in a stage.
    """
    try:
        sql = f"SELECT GET_PRESIGNED_URL('@{DOCS_STAGE}', ?, ?) AS URL"
        url = sql_scalar(sql, params=[path, seconds])
        return url
    except Exception as e:
        st.warning(f"Could not create presigned URL for '{path}': {e}")
        return None

# ---------------------------- SEARCH HELPERS --------------------
def _extract_hit(hit: dict) -> dict:
    row = hit.get("row") or hit or {}
    scores = hit.get("@scores") or hit.get("scores") or {}
    return {
        "DOC_ID":        row.get("DOC_ID"),
        "FILENAME":      row.get("FILENAME"),
        "RELATIVE_PATH": row.get("RELATIVE_PATH"),
        "PERSON":        row.get("PERSON"),
        "DOC_TYPE":      row.get("DOC_TYPE"),
        "DOC_DATE":      row.get("DOC_DATE"),
        # Optional if the service returns them; otherwise None
        "score_sem":     scores.get("cosine_similarity"),
        "score_text":    scores.get("text_match"),
    }

def run_search(query: str, limit: int, service_name: str) -> list[dict]:
    """
    Call SEARCH_PREVIEW with a safe column list (no @scores requirement).
    If scores are present, we still surface them.
    """
    payload = {
        "query": query,
        "limit": int(limit),
        "columns": ["DOC_ID","FILENAME","RELATIVE_PATH","PERSON","DOC_TYPE","DOC_DATE"]
    }
    sql = "SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(?, ?) AS RESULT"
    r = session.sql(sql, params=[service_name, json.dumps(payload)]).collect()
    if not r:
        return []
    raw = r[0]["RESULT"]
    data = json.loads(raw) if isinstance(raw, str) else raw
    results = (data or {}).get("results") or []
    return [_extract_hit(h) for h in results]

def filter_hits_locally(rows, person=None, doc_type=None, d_from=None, d_to=None):
    from datetime import date
    def to_date(s):
        if not s: return None
        try: return date.fromisoformat(str(s)[:10])
        except: return None

    out = []
    for r in rows:
        if person and (r.get("PERSON") or "").lower() != person.lower():
            continue
        if doc_type and (r.get("DOC_TYPE") or "").lower() != doc_type.lower():
            continue
        if d_from or d_to:
            dd = to_date(r.get("DOC_DATE"))
            if d_from and (not dd or dd < d_from): continue
            if d_to   and (not dd or dd > d_to):   continue
        out.append(r)
    return out

# ---------------------------- AGENT HELPER ----------------------
def agent_answer(prompt: str) -> str:
    """
    Calls your FA_DOCS_AGENT via the Cortex function and returns text.
    This will gracefully return the error text if the function isn't available.
    """
    try:
        sql = "SELECT SNOWFLAKE.CORTEX.EXECUTE_AGENT(?, OBJECT_CONSTRUCT('input', ?)) AS R"
        res = session.sql(sql, params=["FA_DOCS_AGENT", prompt]).collect()
        if not res:
            return "Agent returned no result."
        obj = res[0]["R"]
        data = json.loads(obj) if isinstance(obj, str) else obj
        if isinstance(data, dict):
            msg = data.get("message")
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
            msgs = data.get("messages") or data.get("output") or []
            if isinstance(msgs, list):
                for m in reversed(msgs):
                    if isinstance(m, dict) and m.get("content"):
                        return m["content"]
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Agent call failed: {e}"

# ----------------------------- TABS -----------------------------
tab1, tab2, tab3 = st.tabs(["🗃️ Data Explorer", "📄 Docs Search", "💬 Chat with Agent"])

# =========================== TAB 1 ==============================
with tab1:
    st.subheader(f"Tables in {DB}.{SCHEMA}")
    tables_sql = f"""
      SELECT t.table_schema, t.table_name, t.row_count,
             COALESCE(m.active_bytes,0)/POWER(1024,2) AS size_mb,
             t.created
      FROM {DB}.INFORMATION_SCHEMA.TABLES t
      LEFT JOIN {DB}.INFORMATION_SCHEMA.TABLE_STORAGE_METRICS m
        ON m.table_schema = t.table_schema AND m.table_name = t.table_name
      WHERE t.table_schema = '{SCHEMA}' AND t.table_type = 'BASE TABLE'
      ORDER BY size_mb DESC, t.table_name;
    """
    tables_df = sql_to_df(tables_sql)
    st.dataframe(tables_df, use_container_width=True, height=280)

    st.divider()
    st.subheader("Preview a table")
    tbl = st.selectbox(
        "Pick a table",
        options=tables_df["TABLE_NAME"].tolist(),
        index=0 if not tables_df.empty else None,
        key="tbl_pick",
    )
    if tbl:
        preview_df = sql_to_df(f"SELECT * FROM {DB}.{SCHEMA}.{tbl} LIMIT {PREVIEW_LIMIT}")
        st.caption(f"Showing up to {PREVIEW_LIMIT} rows from {DB}.{SCHEMA}.{tbl}")
        st.dataframe(preview_df, use_container_width=True)

# =========================== TAB 2 ==============================
with tab2:
    st.subheader("Semantic search over uploaded documents")

    # --- Filters (client-side only) ---
    with st.container(border=True):
        q = st.text_input("Query", placeholder="e.g., remote work policy, onboarding, invoice total …", key="q_text")

        colA, colB, colC, colD = st.columns([1, 1, 1, 1])

        persons = session.sql(f"""
            SELECT DISTINCT PERSON FROM {DB}.{SCHEMA}.RAW_DOCS
            WHERE PERSON IS NOT NULL ORDER BY 1
        """).to_pandas()["PERSON"].tolist()
        persons = ["(any)"] + persons

        doctypes = session.sql(f"""
            SELECT DISTINCT LOWER(DOC_TYPE) AS DOC_TYPE FROM {DB}.{SCHEMA}.RAW_DOCS
            WHERE DOC_TYPE IS NOT NULL ORDER BY 1
        """).to_pandas()["DOC_TYPE"].tolist()
        doctypes = ["(any)"] + doctypes

        person_sel = colA.selectbox("Person (optional)", persons, index=0, key="person_sel")
        dtype_sel  = colB.selectbox("Doc type (optional)", doctypes, index=0, key="dtype_sel")
        d_from     = colC.date_input("From date", value=None, key="d_from")
        d_to       = colD.date_input("To date",   value=None, key="d_to")

        limit = st.slider("Max results", min_value=1, max_value=50, value=25, key="limit_slider")

    btn = st.button("Search", type="primary")

    if btn:
        if not q or not q.strip():
            st.warning("Enter a search query first.")
        else:
            # normalize filters to None when (any) or empty
            p = None if person_sel in (None, "", "(any)") else person_sel
            t = None if dtype_sel  in (None, "", "(any)") else dtype_sel
            d1 = d_from if isinstance(d_from, dt.date) else None
            d2 = d_to   if isinstance(d_to,   dt.date) else None

            try:
                # Always client-side: SEARCH_PREVIEW + local filters
                rows = run_search(q.strip(), int(limit), SEARCH_SERVICE)
                rows = filter_hits_locally(rows, person=p, doc_type=t, d_from=d1, d_to=d2)
                hits_df = pd.DataFrame(rows)
            except Exception as e:
                st.error(f"Search failed: {e}")
                hits_df = pd.DataFrame()

            if hits_df.empty:
                st.info("No matches.")
            else:
                cols = ["DOC_ID","FILENAME","RELATIVE_PATH","PERSON","DOC_TYPE","DOC_DATE","score_sem","score_text"]
                hits_df = hits_df[[c for c in cols if c in hits_df.columns]]
                st.dataframe(hits_df, use_container_width=True, height=350)

                st.caption("Preview (requires files in a stage you can presign).")
                if "RELATIVE_PATH" in hits_df and not hits_df["RELATIVE_PATH"].isna().all():
                    sel = st.selectbox(
                        "Pick a path to preview",
                        hits_df["RELATIVE_PATH"].dropna().unique(),
                        key="preview_path",
                    )
                    if sel:
                        # url = get_presigned_url(sel, PREVIEW_SECONDS)
                        # if url:
                        #     st.components.v1.iframe(url, height=600, scrolling=True)
                        pass

# =========================== TAB 3 ==============================
with tab3:
    st.subheader("Chat with Agent")
    st.caption("Using agent:  **FA_DOCS_AGENT**.  Tip: ask for tables, e.g., “count by doc_type last 12 months”.")

    user_q = st.text_area(
        "Your question",
        height=120,
        placeholder="Examples: How many documents are in RAW_DOCS?  •  List invoices in 2025.  •  Search policies about remote work and summarize."
    )

    ask = st.button("Ask Agent", type="primary", key="ask_agent_btn")

    if ask:
        if not user_q or not user_q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking…"):
                ans = agent_answer(user_q.strip())
            st.write(ans)
