import io
import os
from datetime import date, timedelta, datetime

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ================= App Config =================
st.set_page_config(page_title="KRW 환율 그래프 (ECOS)", page_icon="💱", layout="wide", initial_sidebar_state="expanded")
APP_TITLE = "💱 해외통화 대비 원화 환율 그래프 (한국은행 ECOS)"
ECOS_TABLE_CODE = "731Y001"  # 3.1.1.1. 주요국 통화의 대원화환율 (주기 D)
MAX_ROWS = 100000

CURRENCIES = {
    "USD": "미국 달러 (USD)",
    "EUR": "유로 (EUR)",
    "JPY": "일본 엔 (JPY)",
    "CNY": "중국 위안 (CNY)",
    "GBP": "영국 파운드 (GBP)",
    "AUD": "호주 달러 (AUD)",
    "CAD": "캐나다 달러 (CAD)",
    "CHF": "스위스 프랑 (CHF)",
    "HKD": "홍콩 달러 (HKD)",
    "SGD": "싱가포르 달러 (SGD)",
    "NZD": "뉴질랜드 달러 (NZD)",
}

# JPY는 100엔 기준 제공 → 1엔 기준으로 환산
PER_UNIT_DEFAULT = {"JPY": 100}

# ================= ECOS helpers =================
def get_ecos_key() -> str:
    # 1) Streamlit Cloud secrets
    try:
        k = st.secrets.get("ECOS_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    # 2) Env var
    k = os.getenv("ECOS_API_KEY", "")
    if k:
        return k
    # 3) Hardcoded (local test only; do NOT commit real keys)
    return "YOUR_ECOS_API_KEY"

def ecos_get_item_list(diag=None) -> list[dict]:
    key = get_ecos_key()
    url = f"https://ecos.bok.or.kr/api/StatisticItemList/{key}/json/kr/1/{MAX_ROWS}/{ECOS_TABLE_CODE}/"
    if diag is not None:
        diag.append(f"ECOS ItemList: {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    block = data.get("StatisticItemList")
    if not block or "row" not in block:
        result = data.get("RESULT", {})
        code = (result or {}).get("CODE", "UNKNOWN")
        msg  = (result or {}).get("MESSAGE", "ECOS API 응답 오류")
        raise ValueError(f"ECOS API 오류: {code} {msg}")
    rows = block.get("row") or []
    if not rows:
        raise ValueError("통화 항목 목록이 비어 있습니다.")
    return rows

def ecos_resolve_item_codes(diag=None) -> dict:
    rows = ecos_get_item_list(diag=diag)
    alias_map = {
        "USD": ["미국 달러","USD"],
        "EUR": ["유로","EUR"],
        "JPY": ["일본 엔","JPY"],
        "CNY": ["중국 위안","CNY"],
        "GBP": ["영국 파운드","GBP"],
        "AUD": ["호주 달러","AUD"],
        "CAD": ["캐나다 달러","CAD"],
        "CHF": ["스위스 프랑","CHF"],
        "HKD": ["홍콩 달러","HKD"],
        "SGD": ["싱가포르 달러","SGD"],
        "NZD": ["뉴질랜드 달러","NZD"],
    }
    comp = {}
    for row in rows:
        code = str(row.get("ITEM_CODE1","")).strip()
        name = str(row.get("ITEM_NAME1","")).strip()
        if not code or not name:
            continue
        cur_key = None
        for k, aliases in alias_map.items():
            if any(a.lower() in name.lower() for a in aliases):
                cur_key = k
                break
        if cur_key:
            comp[cur_key] = code
    if diag is not None:
        diag.append(f"Resolved item codes: {comp}")
    return comp

def ecos_timeseries(item_code: str, start_yyyymmdd: str, end_yyyymmdd: str, diag=None) -> pd.DataFrame:
    key = get_ecos_key()
    # 731Y001은 ITEM_CODE1만 필요. 부족한 하위항목은 ?로 채워도 허용.
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/{MAX_ROWS}/"
        f"{ECOS_TABLE_CODE}/D/{start_yyyymmdd}/{end_yyyymmdd}/{item_code}/?/?/?/"
    )
    if diag is not None:
        diag.append(f"ECOS Search: {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    block = data.get("StatisticSearch")
    if not block or "row" not in block:
        result = data.get("RESULT", {})
        code = (result or {}).get("CODE", "UNKNOWN")
        msg  = (result or {}).get("MESSAGE", "ECOS API 응답 오류")
        raise ValueError(f"ECOS API 오류: {code} {msg}")
    rows = block.get("row") or []
    if not rows:
        raise ValueError("요청 구간 데이터 없음 (INFO-200)")
    recs = []
    for obj in rows:
        t = str(obj.get("TIME","")).strip()
        v = str(obj.get("DATA_VALUE","")).strip()
        if not t or not v:
            continue
        dt = pd.to_datetime(t, errors="coerce")
        if pd.isna(dt):
            continue
        try:
            val = float(v.replace(",",""))
        except Exception:
            continue
        recs.append({"date": dt, "value": val})
    if not recs:
        raise ValueError("구문은 정상이나 변환 가능한 데이터가 없습니다.")
    df = pd.DataFrame(recs).sort_values("date").set_index("date")
    return df

def fetch_series(currency: str, start_date: date, end_date: date, diag=None) -> pd.DataFrame:
    mapping = ecos_resolve_item_codes(diag=diag)
    item_code = mapping.get(currency)
    if not item_code:
        raise ValueError(f"{currency}의 아이템코드를 찾지 못했습니다. (ECOS {ECOS_TABLE_CODE})")
    s = start_date.strftime("%Y%m%d")
    e = end_date.strftime("%Y%m%d")
    raw = ecos_timeseries(item_code, s, e, diag=diag)
    rng = pd.date_range(start=start_date, end=end_date, freq="D")
    df = raw.reindex(rng).ffill()
    df.index.name = "date"
    per_unit = PER_UNIT_DEFAULT.get(currency, 1)
    df["value"] = df["value"] / (per_unit or 1)
    return df

def last_available_rate(df: pd.DataFrame, end_dt: pd.Timestamp):
    if end_dt in df.index:
        return end_dt, float(df.loc[end_dt, "value"])
    prev = df.index[df.index <= end_dt]
    if len(prev) == 0:
        return df.index.min(), float(df.iloc[0]["value"])
    last_idx = prev.max()
    return last_idx, float(df.loc[last_idx, "value"])

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, encoding="utf-8")
    return buf.getvalue().encode("utf-8")

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "FX") -> bytes:
    out = io.BytesIO()
    try:
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    except Exception:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    out.seek(0)
    return out.read()

# ================= UI =================
st.title(APP_TITLE)
st.caption("데이터 소스: 한국은행 ECOS OpenAPI — 3.1.1.1. 주요국 통화의 대원화환율(731Y001, 주기 D). "
           "JPY는 100엔 기준 제공이므로 1엔 기준으로 환산하여 표기합니다.")

with st.sidebar:
    st.header("설정 / 입력")
    cur = st.selectbox("통화 선택", options=list(CURRENCIES.keys()), format_func=lambda k: CURRENCIES[k])
    today = date.today()
    start_default = today - timedelta(days=90)
    start_dt = st.date_input("시작일", start_default, max_value=today)
    end_dt = st.date_input("종료일", today, min_value=start_dt, max_value=today)
    show_diag = st.checkbox("진단(요청/응답 로그) 보기", value=True)
    run = st.button("그래프 그리기", type="primary")

if run:
    diag = []
    try:
        data = fetch_series(cur, start_dt, end_dt, diag=diag)

        if show_diag:
            with st.expander("진단(요청 URL 및 상태)"):
                for line in diag:
                    st.code(line)

        if data.empty:
            st.warning("표시할 데이터가 없습니다. 기간을 넓혀 보거나 다른 통화를 선택해 보세요.")
            st.stop()

        end_ts = pd.to_datetime(end_dt.isoformat())
        last_dt, rate = last_available_rate(data, end_ts)
        st.markdown(
            f'''<div style="padding:12px 16px;border-radius:12px;background:#f5f5f5;font-size:18px;">
            <b>최종 환율</b> — {cur}: {rate:,.4f} KRW (기준일 {last_dt.date().isoformat()})
            </div>''',
            unsafe_allow_html=True,
        )

        st.subheader(f"일별 환율 추이 (KRW per 1 {cur})")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(data.index, data["value"], label=f"{cur}/KRW")
        ax.set_xlabel("날짜")
        ax.set_ylabel(f"KRW per 1 {cur}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        st.subheader("데이터 표")
        disp = data.rename(columns={"value": f"{cur}/KRW"}).round(6)
        st.dataframe(disp)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "CSV 다운로드",
                data=to_csv_bytes(disp),
                file_name=f"krw_fx_{cur}_{start_dt.isoformat()}_{end_dt.isoformat()}.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                "엑셀(.xlsx) 다운로드",
                data=to_excel_bytes(disp),
                file_name=f"krw_fx_{cur}_{start_dt.isoformat()}_{end_dt.isoformat()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        if show_diag and diag:
            with st.expander("진단(요청 URL 및 상태)"):
                for line in diag:
                    st.code(line)
        st.error(f"오류: {e}")

st.markdown("---")
st.caption("ECOS 731Y001(주요국 통화의 대원화환율, D) 기준. 주말/공휴일 공백은 전일값으로 보정합니다.")
