import io
import os
from datetime import date, timedelta, datetime

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title='KRW 환율 그래프 (ECOS)', page_icon='💱', layout='wide', initial_sidebar_state='expanded')
APP_TITLE = '💱 해외통화 대비 원화 환율 그래프 (한국은행 ECOS)'
ECOS_TABLE_CODE = '731Y001'  # 3.1.1.1. 주요국 통화의 대원화환율 (주기 D)
MAX_ROWS = 100000

CURRENCIES = {
    'USD': '미국 달러 (USD)', 'EUR': '유로 (EUR)', 'JPY': '일본 엔 (JPY)', 'CNY': '중국 위안 (CNY)',
    'GBP': '영국 파운드 (GBP)', 'AUD': '호주 달러 (AUD)', 'CAD': '캐나다 달러 (CAD)', 'CHF': '스위스 프랑 (CHF)',
    'HKD': '홍콩 달러 (HKD)', 'SGD': '싱가포르 달러 (SGD)', 'NZD': '뉴질랜드 달러 (NZD)'
}
PER_UNIT_DEFAULT = {'JPY': 100}

def get_ecos_key() -> str:
    try:
        k = st.secrets.get('ECOS_API_KEY', '')
        if k: return k
    except Exception:
        pass
    k = os.getenv('ECOS_API_KEY', '')
    if k: return k
    return 'YOUR_ECOS_API_KEY'

def ecos_get_item_list(diag=None) -> list[dict]:
    key = get_ecos_key()
    url = f'https://ecos.bok.or.kr/api/StatisticItemList/{key}/json/kr/1/{MAX_ROWS}/{ECOS_TABLE_CODE}/'
    if diag is not None: diag.append(f'ECOS ItemList: {url}')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    block = data.get('StatisticItemList')
    if not block or 'row' not in block:
        result = data.get('RESULT', {})
        code = (result or {}).get('CODE', 'UNKNOWN')
        msg  = (result or {}).get('MESSAGE', 'ECOS API 응답 오류')
        raise ValueError(f'ECOS API 오류: {code} {msg}')
    rows = block.get('row') or []
    if not rows:
        raise ValueError('통화 항목 목록이 비어 있습니다.')
    return rows

def _collect_names(row: dict) -> str:
    names = []
    for k in ['ITEM_NAME', 'ITEM_NAME1', 'ITEM_NAME2', 'ITEM_NAME3', 'ITEM_NAME4']:
        v = str(row.get(k, '')).strip()
        if v:
            names.append(v)
    return ' | '.join(names)

def ecos_resolve_item_codes(diag=None) -> dict:
    rows = ecos_get_item_list(diag=diag)
    comp = {}

    def which_currency(label: str) -> str | None:
        L = label.lower()
        if ('미국' in L and '달러' in L) or 'usd' in L: return 'USD'
        if '유로' in L or 'eur' in L: return 'EUR'
        if ('일본' in L and '엔' in L) or 'jpy' in L: return 'JPY'
        if ('중국' in L and '위안' in L) or 'cny' in L: return 'CNY'
        if ('영국' in L and '파운드' in L) or 'gbp' in L: return 'GBP'
        if ('호주' in L and '달러' in L) or 'aud' in L: return 'AUD'
        if ('캐나다' in L and '달러' in L) or 'cad' in L: return 'CAD'
        if ('스위스' in L and '프랑' in L) or 'chf' in L: return 'CHF'
        if ('홍콩' in L and '달러' in L) or 'hkd' in L: return 'HKD'
        if ('싱가포르' in L and '달러' in L) or 'sgd' in L: return 'SGD'
        if ('뉴질랜드' in L and '달러' in L) or 'nzd' in L: return 'NZD'
        return None

    samples = []
    for row in rows:
        code = str(row.get('ITEM_CODE1', '') or row.get('ITEM_CODE', '')).strip()
        label = _collect_names(row)
        if len(samples) < 12:
            samples.append(f"{code} :: {label}")
        if not code or not label:
            continue
        cur = which_currency(label)
        if cur and cur not in comp:
            comp[cur] = code

    if diag is not None:
        diag.append(f'Resolved item codes: {comp}')
        if not comp:
            diag.append('ItemList 예시(12개까지):')
            for s in samples:
                diag.append(f'  - {s}')
    return comp

def ecos_timeseries(item_code: str, start_yyyymmdd: str, end_yyyymmdd: str, diag=None) -> pd.DataFrame:
    key = get_ecos_key()
    url = (
        f'https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/{MAX_ROWS}/'
        f'{ECOS_TABLE_CODE}/D/{start_yyyymmdd}/{end_yyyymmdd}/{item_code}/?/?/?/'
    )
    if diag is not None: diag.append(f'ECOS Search: {url}')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    block = data.get('StatisticSearch')
    if not block or 'row' not in block:
        result = data.get('RESULT', {})
        code = (result or {}).get('CODE', 'UNKNOWN')
        msg  = (result or {}).get('MESSAGE', 'ECOS API 응답 오류')
        raise ValueError(f'ECOS API 오류: {code} {msg}')
    rows = block.get('row') or []
    if not rows:
        raise ValueError('요청 구간 데이터 없음 (INFO-200)')
    recs = []
    for obj in rows:
        t = str(obj.get('TIME','')).strip()
        v = str(obj.get('DATA_VALUE','')).strip()
        if not t or not v:
            continue
        dt = pd.to_datetime(t, errors='coerce')
        if pd.isna(dt):
            continue
        try:
            val = float(v.replace(',',''))
        except Exception:
            continue
        recs.append({'date': dt, 'value': val})
    if not recs:
        raise ValueError('구문은 정상이나 변환 가능한 데이터가 없습니다.')
    df = pd.DataFrame(recs).sort_values('date').set_index('date')
    return df

def parse_manual_map(s: str) -> dict:
    m = {}
    s = (s or '').strip()
    if not s:
        return m
    import re
    parts = re.split(r'[\n,]+', s)
    for p in parts:
        if '=' in p:
            k, v = p.split('=', 1)
            m[k.strip().upper()] = v.strip()
    return m

def fetch_series(currency: str, start_date: date, end_date: date, manual: dict, diag=None) -> pd.DataFrame:
    # 1) 수동 매핑 우선
    item_code = (manual or {}).get(currency)
    mapping_used = 'manual' if item_code else 'auto'
    if not item_code:
        mapping = ecos_resolve_item_codes(diag=diag)
        item_code = mapping.get(currency)
    if not item_code:
        raise ValueError(f'{currency}의 아이템코드를 찾지 못했습니다. 수동 매핑을 입력해 주세요 (예: USD=0000001).')

    if diag is not None:
        diag.append(f'Using {mapping_used} item_code for {currency}: {item_code}')

    s = start_date.strftime('%Y%m%d')
    e = end_date.strftime('%Y%m%d')
    raw = ecos_timeseries(item_code, s, e, diag=diag)
    rng = pd.date_range(start=start_date, end=end_date, freq='D')
    df = raw.reindex(rng).ffill()
    df.index.name = 'date'
    per_unit = PER_UNIT_DEFAULT.get(currency, 1)
    df['value'] = df['value'] / (per_unit or 1)
    return df

def last_available_rate(df: pd.DataFrame, end_dt: pd.Timestamp):
    if end_dt in df.index:
        return end_dt, float(df.loc[end_dt, 'value'])
    prev = df.index[df.index <= end_dt]
    if len(prev) == 0:
        return df.index.min(), float(df.iloc[0]['value'])
    last_idx = prev.max()
    return last_idx, float(df.loc[last_idx, 'value'])

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, encoding='utf-8')
    return buf.getvalue().encode('utf-8')

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = 'FX') -> bytes:
    out = io.BytesIO()
    try:
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    except Exception:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    out.seek(0)
    return out.read()

# ================= UI =================
st.title(APP_TITLE)
st.caption('데이터 소스: 한국은행 ECOS — 3.1.1.1. 주요국 통화의 대원화환율(731Y001, 주기 D). JPY는 100엔 기준 제공으로 1엔 기준으로 환산하여 표기합니다.')

with st.sidebar:
    st.header('설정 / 입력')
    cur = st.selectbox('통화 선택', options=list(CURRENCIES.keys()), format_func=lambda k: CURRENCIES[k])
    today = date.today()
    start_default = today - timedelta(days=90)
    start_dt = st.date_input('시작일', start_default, max_value=today)
    end_dt = st.date_input('종료일', today, min_value=start_dt, max_value=today)
    manual_map_text = st.text_input('수동 아이템코드(선택)', placeholder='예: USD=0000001, EUR=0000003')
    show_diag = st.checkbox('진단(요청/응답 로그) 보기', value=True)
    run = st.button('그래프 그리기', type='primary')

if run:
    diag = []
    try:
        manual = parse_manual_map(manual_map_text)
        if manual:
            diag.append(f'Manual map parsed: {manual}')

        data = fetch_series(cur, start_dt, end_dt, manual, diag=diag)

        if show_diag:
            with st.expander('진단(요청 URL 및 상태)'):
                for line in diag:
                    st.code(line)

        if data.empty:
            st.warning('표시할 데이터가 없습니다. 기간을 넓혀 보거나 다른 통화를 선택해 보세요.')
            st.stop()

        end_ts = pd.to_datetime(end_dt.isoformat())
        last_dt, rate = last_available_rate(data, end_ts)
        st.markdown(
            f'''<div style="padding:12px 16px;border-radius:12px;background:#f5f5f5;font-size:18px;">
            <b>최종 환율</b> — {cur}: {rate:,.4f} KRW (기준일 {last_dt.date().isoformat()})
            </div>''',
            unsafe_allow_html=True,
        )

        st.subheader(f'일별 환율 추이 (KRW per 1 {cur})')
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(data.index, data['value'], label=f'{cur}/KRW')
        ax.set_xlabel('날짜')
        ax.set_ylabel(f'KRW per 1 {cur}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        st.subheader('데이터 표')
        disp = data.rename(columns={'value': f'{cur}/KRW'}).round(6)
        st.dataframe(disp)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button('CSV 다운로드', data=to_csv_bytes(disp), file_name=f'krw_fx_{cur}_{start_dt.isoformat()}_{end_dt.isoformat()}.csv', mime='text/csv')
        with c2:
            st.download_button('엑셀(.xlsx) 다운로드', data=to_excel_bytes(disp), file_name=f'krw_fx_{cur}_{start_dt.isoformat()}_{end_dt.isoformat()}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        if show_diag and diag:
            with st.expander('진단(요청 URL 및 상태)'):
                for line in diag:
                    st.code(line)
        st.error(f'오류: {e}')

st.markdown('---')
st.caption('ECOS 731Y001 기준. 자동 매핑이 실패하면 수동 아이템코드를 입력해 주세요 (예: USD=0000001).')
