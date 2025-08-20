import io
import os
import re
from datetime import date, timedelta, datetime

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ===== App Config =====
st.set_page_config(page_title="KRW 환율 그래프 (ECOS)", page_icon="💱", layout="wide", initial_sidebar_state="expanded")
APP_TITLE = "💱 해외통화 대비 원화 환율 그래프 (한국은행 ECOS)"
ECOS_TABLE_CODE = "036Y001"   # 일별 매매기준율 (원화대비)
MAX_ROWS = 100000

# 단일 선택 통화 목록
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

# JPY는 ECOS가 100엔 기준으로 제공 → 1엔 기준으로 환산 필요
PER_UNIT_DEFAULT = {"JPY": 100}

# ===== ECOS 키 불러오기 =====
def get_ecos_key() -> str:
    # 1) Streamlit Secrets
    try:
        k = st.secrets.get("ECOS_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    # 2) 환경변수
    k = os.getenv("ECOS_API_KEY", "")
    if k:
        return k
    # 3) 코드에 직입 (공개 저장소에 올리면 노출 위험!)
    return "YOUR_ECOS_API_KEY"  # ← 로컬 테스트 시 본인 키로 교체하세요.

# ===== ECOS 헬퍼 =====
def ecos_get_item_list(stat_code: str, diag: list | None = None) -> list[dict]:
    key = get_ecos_key()
    url = f"https://ecos.bok.or.kr/api/StatisticItemList/{key}/json/kr/1/{MAX_ROWS}/{stat_code}/"
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


def ecos_resolve_item_codes_036Y001(diag: list | None = None) -> dict:
    """
    036Y001 전용: ITEM_CODE1(통화) + 필요 시 ITEM_CODE2(세부항목: '매매기준율')을 합쳐 "code1/code2"로 반환.
    예: {'USD': '0000001/0000001', 'JPY': '0000002/0000001', ...}
    """
    rows = ecos_get_item_list(ECOS_TABLE_CODE, diag=diag)

    alias_map = {
        "USD": ["미국 달러", "USD"],
        "EUR": ["유로", "EUR"],
        "JPY": ["일본 엔", "JPY"],
        "CNY": ["중국 위안", "CNY"],
        "GBP": ["영국 파운드", "GBP"],
        "AUD": ["호주 달러", "AUD"],
        "CAD": ["캐나다 달러", "CAD"],
        "CHF": ["스위스 프랑", "CHF"],
        "HKD": ["홍콩 달러", "HKD"],
        "SGD": ["싱가포르 달러", "SGD"],
        "NZD": ["뉴질랜드 달러", "NZD"],
    }

    comp: dict[str, str] = {}
    for row in rows:
        c1 = str(row.get("ITEM_CODE1", "")).strip()
        n1 = str(row.get("ITEM_NAME1", "")).strip()
        c2 = str(row.get("ITEM_CODE2", "")).strip()
        n2 = str(row.get("ITEM_NAME2", "")).strip()
        if not c1 or not n1:
            continue

        # 어떤 통화인지 식별
        cur_key = None
        for k, aliases in alias_map.items():
            if any(a.lower() in n1.lower() for a in aliases):
                cur_key = k
                break
        if not cur_key:
            continue

        # 세부항목에 '매매기준'이 들어가면 code1/code2 사용, 아니면 code1만
        if n2 and ("매매기준" in n2):
            comp[cur_key] = f"{c1}/{c2}" if c2 else c1
        elif not n2 and ("매매기준" in n1):
            comp[cur_key] = c1

    if diag is not None:
        diag.append(f"Resolved item codes: {comp}")
    return comp


def ecos_timeseries(item_code: str, start_yyyymmdd: str, end_yyyymmdd: str, backfill_days: int = 30, diag: list | None = None) -> pd.DataFrame:
    """
    ECOS StatisticSearch 호출 → 날짜/값 DataFrame
    - CYCLE 'DD' 실패 시 'D' 재시도
    - 데이터 없으면 시작일을 backfill_days만큼 과거로 당겨 재시도
    - item_code에 'code1/code2' 형태 허용
    """
    key = get_ecos_key()

    def _try_once(cycle: str, s: str, e: str) -> pd.DataFrame | None:
        parts = [ECOS_TABLE_CODE, cycle, s, e] + [p for p in str(item_code).split("/") if p]
        url = f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/{MAX_ROWS}/" + "/".join(parts) + "/"
        if diag is not None:
            diag.append(f"ECOS Search: {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        block = data.get("StatisticSearch")
        if not block or "row" not in block:
            return None
        rows = block.get("row") or []
        if not rows:
            return None

        recs = []
        for obj in rows:
            t = str(obj.get("TIME", "")).strip()
            v = str(obj.get("DATA_VALUE", "")).strip()
            if not t or not v:
                continue
            dt = pd.to_datetime(t, errors="coerce")
            if pd.isna(dt):
                # 'YYYYMM'만 오는 표 대비, 마지막날로 보정 가능(036Y001은 일별이라 보통 필요 없음)
                continue
            try:
                val = float(v.replace(",", ""))
            except Exception:
                continue
            recs.append({"date": dt, "value": val})
        if not recs:
            return None
        df = pd.DataFrame(recs).sort_values("date").set_index("date")
        return df

    # 1) 원래 기간 시도
    for cyc in ("DD", "D"):
        df = _try_once(cyc, start_yyyymmdd, end_yyyymmdd)
        if df is not None:
            return df

    # 2) 백필 재시도
    sdt = datetime.strptime(start_yyyymmdd, "%Y%m%d").date()
    bf = (sdt - timedelta(days=backfill_days)).strftime("%Y%m%d")
    for cyc in ("DD", "D"):
        df = _try_once(cyc, bf, end_yyyymmdd)
        if df is not None:
            return df

    raise ValueError("ECOS 무자료(INFO-200) 또는 조건 미일치로 데이터 수신 실패")


def fetch_series(currency: str, start_date: date, end_date: date, backfill_days: int, diag: list | None = None) -> pd.DataFrame:
    # 1) 통화별 per_unit (예: JPY=100)
    per_unit = PER_UNIT_DEFAULT.get(currency, 1)

    # 2) 복합 코드 자동 탐지
    comp = ecos_resolve_item_codes_036Y001(diag=diag)

    # 3) 통화→아이템코드 결정
    item_code = comp.get(currency)
    if not item_code:
        # 기본 가드(일부 표에서 흔히 쓰는 코드1), 필요 시 수동 입력으로 덮어쓸 수 있게 함
        defaults = {"USD": "0000001", "JPY": "0000002", "EUR": "0000003", "CNY": "0000004"}
        item_code = defaults.get(currency)
        if item_code is None:
            raise ValueError(f"{currency}에 대한 ECOS 아이템코드를 자동으로 찾지 못했습니다. 사이드바 수동 매핑을 이용하세요.")

    # 4) 시계열 조회
    s = start_date.strftime("%Y%m%d")
    e = end_date.strftime("%Y%m%d")
    raw = ecos_timeseries(item_code, s, e, backfill_days=backfill_days, diag=diag)

    # 5) 요청 구간으로 슬라이스 + 결측 보정(전일 값 유지)
    rng = pd.date_range(start=start_date, end=end_date, freq="D")
    df = raw.reindex(rng).ffill()
    df.index.name = "date"

    # 6) JPY 100엔→1엔 환산 등 per_unit 적용
    df["value"] = df["value"] / (per_unit or 1)
    return df


def last_available_rate(df: pd.DataFrame, end_dt: pd.Timestamp) -> tuple[pd.Timestamp, float]:
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

# ===== UI =====
st.title(APP_TITLE)
st.caption("데이터 소스: 한국은행 ECOS OpenAPI(036Y001, 일별 매매기준율). "
           "JPY는 100엔 기준 제공 → 1엔 기준으로 환산하여 표기합니다.")

with st.sidebar:
    st.header("설정 / 입력")
    cur = st.selectbox("통화 선택", options=list(CURRENCIES.keys()), format_func=lambda k: CURRENCIES[k])

    today = date.today()
    start_default = today - timedelta(days=90)
    start_dt = st.date_input("시작일", start_default, max_value=today)
    end_dt = st.date_input("종료일", today, min_value=start_dt, max_value=today)

    backfill_days = st.slider("백필(과거로 추가 조회) 일수", min_value=0, max_value=90, value=30,
                              help="요청 구간에 데이터가 없을 때 시작일을 과거로 당겨 재조회합니다.")

    manual_map = st.text_input("수동 아이템코드(선택)", placeholder="예: USD=0000001/0000001")
    show_diag = st.checkbox("진단(요청/응답 로그) 보기", value=False)

    run = st.button("그래프 그리기", type="primary")

# 수동 매핑 파서
def parse_manual_map(s: str) -> dict:
    m = {}
    if not s.strip():
        return m
    parts = re.split(r"[,\n]+", s.strip())
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            m[k.strip().upper()] = v.strip()
    return m

if run:
    diag: list[str] = []
    try:
        # 수동 매핑 우선 적용
        manual = parse_manual_map(manual_map)
        if manual:
            diag.append(f"Manual map: {manual}")

        # fetch
        # (수동 매핑이 있으면 해당 통화만 강제 사용하도록 임시 덮어쓰기)
        def fetch_with_optional_manual(currency: str) -> pd.DataFrame:
            if currency in manual and manual[currency]:
                # 수동 코드 사용
                s = start_dt.strftime("%Y%m%d")
                e = end_dt.strftime("%Y%m%d")
                raw = ecos_timeseries(manual[currency], s, e, backfill_days=backfill_days, diag=diag)
                rng = pd.date_range(start=start_dt, end=end_dt, freq="D")
                df = raw.reindex(rng).ffill()
                df.index.name = "date"
                per_unit = PER_UNIT_DEFAULT.get(currency, 1)
                df["value"] = df["value"] / (per_unit or 1)
                return df
            # 자동 탐지 경로
            return fetch_series(currency, start_dt, end_dt, backfill_days=backfill_days, diag=diag)

        data = fetch_with_optional_manual(cur)

        if show_diag:
            with st.expander("진단(요청 URL 및 상태)"):
                for line in diag:
                    st.code(line)

        if data.empty:
            st.warning("표시할 데이터가 없습니다. 기간을 넓혀 보거나, 수동 아이템코드를 입력해 보세요.")
            st.stop()

        # 최종 환율 배너
        end_ts = pd.to_datetime(end_dt.isoformat())
        last_dt, rate = last_available_rate(data, end_ts)
        st.markdown(
            f"""<div style="padding:12px 16px;border-radius:12px;background:#f5f5f5;font-size:18px;">
            <b>최종 환율</b> — {cur}: {rate:,.4f} KRW (기준일 {last_dt.date().isoformat()})
            </div>""",
            unsafe_allow_html=True,
        )

        # 그래프
        st.subheader(f"일별 환율 추이 (KRW per 1 {cur})")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(data.index, data["value"], label=f"{cur}/KRW")
        ax.set_xlabel("날짜")
        ax.set_ylabel(f"KRW per 1 {cur}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        # 표 & 다운로드
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
st.caption("출처: 한국은행 ECOS OpenAPI (036Y001, 일별 매매기준율). "
           "ECOS가 JPY를 100엔 기준으로 제공하기 때문에 1엔 기준으로 환산하여 표기합니다. "
           "주말/공휴일 등 공시 공백은 전일 값으로 보정(FFill)하여 시각화합니다.")
