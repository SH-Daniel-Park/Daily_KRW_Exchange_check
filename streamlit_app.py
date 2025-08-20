
import io
import os
import re
from datetime import date, timedelta, datetime

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ================= App Config =================
st.set_page_config(page_title="KRW 환율 비교 (ECOS vs 은행)", page_icon="💱", layout="wide", initial_sidebar_state="expanded")
APP_TITLE = "💱 해외통화 대비 원화 환율 비교 (ECOS vs 은행 고시)"

# ECOS: 3.1.1.1. 주요국 통화의 대원화환율 (주기 D)
ECOS_TABLE_CODE = "731Y001"
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

# JPY는 100엔 기준 제공 → 1엔으로 환산
PER_UNIT_DEFAULT = {"JPY": 100}

SOURCE_OPTIONS = ["ECOS", "은행 TTM(매매기준)", "은행 TTS(팔 때)", "은행 TTB(살 때)"]

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

def _collect_names(row: dict) -> str:
    names = []
    for k in ["ITEM_NAME", "ITEM_NAME1", "ITEM_NAME2", "ITEM_NAME3", "ITEM_NAME4"]:
        v = str(row.get(k, "")).strip()
        if v:
            names.append(v)
    return " | ".join(names)

def ecos_resolve_item_codes(diag=None) -> dict:
    rows = ecos_get_item_list(diag=diag)
    comp = {}

    def which_currency(label: str) -> str | None:
        L = label.lower()
        if ("미국" in L and "달러" in L) or "usd" in L: return "USD"
        if ("유로" in L) or "eur" in L: return "EUR"
        if ("일본" in L and "엔" in L) or "jpy" in L: return "JPY"
        if ("중국" in L and "위안" in L) or "cny" in L: return "CNY"
        if ("영국" in L and "파운드" in L) or "gbp" in L: return "GBP"
        if ("호주" in L and "달러" in L) or "aud" in L: return "AUD"
        if ("캐나다" in L and "달러" in L) or "cad" in L: return "CAD"
        if ("스위스" in L and "프랑" in L) or "chf" in L: return "CHF"
        if ("홍콩" in L and "달러" in L) or "hkd" in L: return "HKD"
        if ("싱가포르" in L and "달러" in L) or "sgd" in L: return "SGD"
        if ("뉴질랜드" in L and "달러" in L) or "nzd" in L: return "NZD"
        return None

    samples = []
    for row in rows:
        code = str(row.get("ITEM_CODE1", "") or row.get("ITEM_CODE", "")).strip()
        label = _collect_names(row)
        if len(samples) < 12:
            samples.append(f"{code} :: {label}")
        if not code or not label:
            continue
        cur = which_currency(label)
        if cur and cur not in comp:
            comp[cur] = code

    if diag is not None:
        diag.append(f"Resolved item codes: {comp}")
        if not comp:
            diag.append("ItemList 예시(최대 12):")
            for s in samples:
                diag.append(f"  - {s}")
    return comp

def ecos_timeseries(item_code: str, start_yyyymmdd: str, end_yyyymmdd: str, diag=None) -> pd.DataFrame:
    key = get_ecos_key()
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

def fetch_ecos_series(currency: str, start_date: date, end_date: date, diag=None) -> pd.DataFrame:
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
    df.rename(columns={"value": "ECOS"}, inplace=True)
    return df

# ================= Bank CSV helpers =================
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]", "", s.lower())

def _guess_date_col(cols):
    for c in cols:
        n = _norm(str(c))
        if any(k in n for k in ["date","날짜","일자","기준일"]):
            return c
    return cols[0]

def _guess_value_col(cols, kind: str):
    kind = kind.lower()
    for c in cols:
        n = _norm(str(c))
        if kind == "ttm" and ("ttm" in n or "기준" in n or "매매기준" in n):
            return c
        if kind == "tts" and ("tts" in n or "팔때" in n or "보낼때" in n or "매도" in n):
            return c
        if kind == "ttb" and ("ttb" in n or "살때" in n or "받을때" in n or "매입" in n):
            return c
    return None

def read_bank_csv(file_bytes: bytes, encoding_try=("utf-8","cp949")) -> pd.DataFrame:
    last_err = None
    for enc in encoding_try:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def normalize_bank_df(df: pd.DataFrame, date_col, ttm_col, tts_col, ttb_col, unit_divisor: int) -> pd.DataFrame:
    out = pd.DataFrame()
    d = pd.to_datetime(df[date_col], errors="coerce")
    out["date"] = d
    out = out.dropna(subset=["date"]).set_index("date").sort_index()

    def to_num(s):
        try:
            return float(str(s).replace(",","").strip())
        except Exception:
            return None

    if ttm_col:
        out["은행 TTM(매매기준)"] = df[ttm_col].map(to_num) / (unit_divisor or 1)
    if tts_col:
        out["은행 TTS(팔 때)"] = df[tts_col].map(to_num) / (unit_divisor or 1)
    if ttb_col:
        out["은행 TTB(살 때)"] = df[ttb_col].map(to_num) / (unit_divisor or 1)
    out = out.dropna(how="all")
    return out

def last_available_from_series(series: pd.Series, end_ts: pd.Timestamp):
    if series.empty:
        return None, None
    idx = series.index[series.index <= end_ts]
    if len(idx)==0:
        return series.index.min(), float(series.iloc[0])
    last_idx = idx.max()
    return last_idx, float(series.loc[last_idx])

# ================= UI =================
st.title(APP_TITLE)
st.caption("소스: 한국은행 ECOS(731Y001, 일별) + 은행 고시(TTM/TTS/TTB, CSV 업로드). "
           "JPY는 은행/ECOS 모두 100엔 단위일 수 있어 단위를 맞춰 비교합니다.")

with st.sidebar:
    st.header("설정 / 입력")
    cur = st.selectbox("통화 선택", options=list(CURRENCIES.keys()), format_func=lambda k: CURRENCIES[k])
    today = date.today()
    start_default = today - timedelta(days=90)
    start_dt = st.date_input("시작일", start_default, max_value=today)
    end_dt = st.date_input("종료일", today, min_value=start_dt, max_value=today)

    sources = st.multiselect("소스 선택(비교선)", options=["ECOS","은행 TTM(매매기준)","은행 TTS(팔 때)","은행 TTB(살 때)"], default=["ECOS"])

    st.markdown("**은행 고시 환율 CSV 업로드 (선택)**")
    bank_file = st.file_uploader("CSV 파일 선택", type=["csv"], help="열 예시: Date, TTM(매매기준), TTS(팔 때), TTB(살 때)")

    bank_unit_default = 100 if cur == "JPY" else 1
    bank_unit = st.radio("은행 데이터 단위", options=[1,100], index=(1 if bank_unit_default==100 else 0),
                         help="은행 CSV가 100JPY 단위면 100을 선택하세요. (다른 통화는 보통 1)")

    ttm_col = tts_col = ttb_col = date_col = None
    bank_df_raw = None
    if bank_file is not None:
        try:
            bank_df_raw = read_bank_csv(bank_file.getvalue())
            cols = list(bank_df_raw.columns)
            date_col_guess = _guess_date_col(cols)
            ttm_guess = _guess_value_col(cols, "ttm")
            tts_guess = _guess_value_col(cols, "tts")
            ttb_guess = _guess_value_col(cols, "ttb")
            st.write("CSV 열 매핑:")
            date_col = st.selectbox("날짜 열", options=cols, index=cols.index(date_col_guess) if date_col_guess in cols else 0)
            ttm_col = st.selectbox("TTM(매매기준) 열", options=["(없음)"] + cols,
                                   index=(["(없음)"]+cols).index(ttm_guess) if ttm_guess in (cols) else 0)
            tts_col = st.selectbox("TTS(팔 때) 열", options=["(없음)"] + cols,
                                   index=(["(없음)"]+cols).index(tts_guess) if tts_guess in (cols) else 0)
            ttb_col = st.selectbox("TTB(살 때) 열", options=["(없음)"] + cols,
                                   index=(["(없음)"]+cols).index(ttb_guess) if ttb_guess in (cols) else 0)
            ttm_col = None if ttm_col == "(없음)" else ttm_col
            tts_col = None if tts_col == "(없음)" else tts_col
            ttb_col = None if ttb_col == "(없음)" else ttb_col
        except Exception as e:
            st.error(f"CSV 읽기 실패: {e}")

    run = st.button("그래프 그리기", type="primary")

st.markdown("---")
with st.expander("은행 고시 환율 가져오는 방법(안내)", expanded=False):
    st.markdown(
        """
- **웹 페이지에서 복사/CSV 저장**: 하나은행/다른 시중은행의 환율 표에서 날짜·TTM/TTS/TTB를 내려받아 CSV로 저장해 업로드하세요.
- **직접 CSV 작성**: 아래 **템플릿 CSV**를 내려받아 날짜와 값을 채워 업로드하세요.
- **단위 맞추기**: JPY는 은행/ECOS가 100엔 단위를 쓰기도 하므로, 사이드바 **은행 데이터 단위**에서 1 또는 100을 선택해 주세요.
- **열 이름은 자유**입니다. 업로드 후에 앱에서 **날짜/TTM/TTS/TTB 열을 직접 매핑**할 수 있습니다.
"""
    )
    tmpl_rng = pd.date_range(start=date.today()-timedelta(days=30), end=date.today(), freq="D")
    tmpl = pd.DataFrame({"Date": tmpl_rng.strftime("%Y-%m-%d"), "TTM": "", "TTS": "", "TTB": ""})
    buf = io.StringIO()
    tmpl.to_csv(buf, index=False)
    st.download_button("은행 환율 템플릿 CSV 다운로드", data=buf.getvalue().encode("utf-8"),
                       file_name=f"bank_fx_template.csv", mime="text/csv")

# ================= Run =================
if run:
    diag = []
    series_frames = []

    try:
        # ECOS
        if "ECOS" in sources:
            ecos_df = fetch_ecos_series(cur, start_dt, end_dt, diag=diag)
            series_frames.append(ecos_df[["ECOS"]])

        # Bank CSV
        if any(s in sources for s in ["은행 TTM(매매기준)", "은행 TTS(팔 때)", "은행 TTB(살 때)"]):
            if bank_df_raw is None:
                st.warning("은행 소스를 선택했지만 CSV가 업로드되지 않았습니다. 템플릿으로 작성하거나 웹에서 저장한 CSV를 올려주세요.")
            else:
                bank_norm = normalize_bank_df(bank_df_raw, date_col, ttm_col, tts_col, ttb_col, unit_divisor=bank_unit)
                rng = pd.date_range(start=start_dt, end=end_dt, freq="D")
                bank_norm = bank_norm.reindex(rng).ffill()
                bank_norm.index.name = "date"
                keep_cols = [c for c in bank_norm.columns if c in sources]
                if keep_cols:
                    series_frames.append(bank_norm[keep_cols])

        if not series_frames:
            st.warning("표시할 데이터가 없습니다. ECOS 또는 은행 소스를 선택하고, CSV를 업로드해 주세요.")
            st.stop()

        all_df = pd.concat(series_frames, axis=1).dropna(how="all")
        if all_df.empty:
            st.warning("표시할 데이터가 없습니다. 기간을 조정하거나, CSV 매핑/단위를 확인해 주세요.")
            st.stop()

        end_ts = pd.to_datetime(end_dt.isoformat())
        badges = []
        for col in all_df.columns:
            s = all_df[col].dropna()
            if s.empty: 
                continue
            idx = s.index[s.index <= end_ts]
            last_dt = idx.max() if len(idx)>0 else s.index.max()
            val = float(s.loc[last_dt])
            badges.append(f"<div style='margin:4px 8px;display:inline-block;padding:8px 10px;border-radius:10px;background:#f5f5f5'>{col}: {val:,.4f} KRW (기준일 {last_dt.date().isoformat()})</div>")
        if badges:
            st.markdown("<div>" + " ".join(badges) + "</div>", unsafe_allow_html=True)

        st.subheader(f"일별 환율 추이 (KRW per 1 {cur}) — 선택 소스 비교")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for col in all_df.columns:
            ax.plot(all_df.index, all_df[col], label=col)
        ax.set_xlabel("날짜")
        ax.set_ylabel(f"KRW per 1 {cur}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        st.subheader("데이터 표")
        disp = all_df.round(6)
        st.dataframe(disp)

        def to_csv_bytes(df: pd.DataFrame) -> bytes:
            sio = io.StringIO()
            df.to_csv(sio, encoding="utf-8")
            return sio.getvalue().encode("utf-8")
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

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("CSV 다운로드", data=to_csv_bytes(disp),
                               file_name=f"fx_compare_{cur}_{start_dt.isoformat()}_{end_dt.isoformat()}.csv",
                               mime="text/csv")
        with c2:
            st.download_button("엑셀(.xlsx) 다운로드", data=to_excel_bytes(disp),
                               file_name=f"fx_compare_{cur}_{start_dt.isoformat()}_{end_dt.isoformat()}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with st.expander("진단(요청 URL 및 상태)"):
            for line in diag:
                st.code(line)

    except Exception as e:
        with st.expander("진단(요청 URL 및 상태)"):
            for line in diag:
                st.code(line)
        st.error(f"오류: {e}")
