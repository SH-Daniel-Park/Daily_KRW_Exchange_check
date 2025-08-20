import io
from datetime import date, timedelta

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

st.set_page_config(page_title='KRW 환율 그래프', page_icon='💱', layout='wide', initial_sidebar_state='expanded')

APP_TITLE = '💱 해외통화 대비 원화 환율 그래프'
HOST_API = 'https://api.exchangerate.host/timeseries'

CURRENCIES = {
    'USD': '미국 달러 (USD)',
    'EUR': '유로 (EUR)',
    'JPY': '일본 엔 (JPY)',
    'GBP': '영국 파운드 (GBP)',
    'AUD': '호주 달러 (AUD)',
    'CAD': '캐나다 달러 (CAD)',
    'CHF': '스위스 프랑 (CHF)',
    'CNY': '중국 위안 (CNY)',
    'HKD': '홍콩 달러 (HKD)',
    'SGD': '싱가포르 달러 (SGD)',
    'NZD': '뉴질랜드 달러 (NZD)'
}

def _yyyy_mm_dd(d: date) -> str:
    return d.strftime('%Y-%m-%d')

def _request(url, params, diag):
    try:
        r = requests.get(url, params=params, timeout=30)
        diag.append(f'REQ: {url} params={params} -> HTTP {r.status_code}')
        r.raise_for_status()
        return r.json()
    except Exception as e:
        diag.append(f'ERR: {e}')
        return None

def _host_timeseries_direct(base_code, start, end, diag) -> pd.DataFrame:
    params = {'start_date': start, 'end_date': end, 'base': base_code, 'symbols': 'KRW'}
    data = _request(HOST_API, params, diag)
    rates = (data or {}).get('rates', {}) or {}
    recs = []
    for d, obj in sorted(rates.items()):
        v = (obj or {}).get('KRW')
        if v is not None:
            recs.append({'date': pd.to_datetime(d), 'value': float(v)})
    if not recs:
        return pd.DataFrame({'value': []})
    df = pd.DataFrame(recs).sort_values('date').set_index('date')
    return df

def _host_timeseries_inverse(target_code, start, end, diag) -> pd.DataFrame:
    params = {'start_date': start, 'end_date': end, 'base': 'KRW', 'symbols': target_code}
    data = _request(HOST_API, params, diag)
    rates = (data or {}).get('rates', {}) or {}
    recs = []
    for d, obj in sorted(rates.items()):
        v = (obj or {}).get(target_code)
        if v:
            recs.append({'date': pd.to_datetime(d), 'value': 1.0 / float(v)})
    if not recs:
        return pd.DataFrame({'value': []})
    df = pd.DataFrame(recs).sort_values('date').set_index('date')
    return df

def _yf_pair_symbol(cur: str) -> str:
    return f'{cur}KRW=X'

def _yf_download(symbol: str, start: date, end: date, diag) -> pd.DataFrame:
    if not YF_AVAILABLE:
        diag.append('yfinance not available -> skip')
        return pd.DataFrame({'value': []})
    end_excl = end + timedelta(days=1)
    try:
        df = yf.download(symbol, start=_yyyy_mm_dd(start), end=_yyyy_mm_dd(end_excl), progress=False, auto_adjust=False)
        diag.append(f'YF: download {symbol} -> {0 if df is None else len(df)} rows')
        if df is None or df.empty:
            return pd.DataFrame({'value': []})
        out = df[['Close']].rename(columns={'Close': 'value'})
        out.index = pd.to_datetime(out.index.date)
        out.index.name = 'date'
        return out
    except Exception as e:
        diag.append(f'YF error: {e}')
        return pd.DataFrame({'value': []})

def _yf_cross(cur: str, start: date, end: date, diag) -> pd.DataFrame:
    if not YF_AVAILABLE:
        return pd.DataFrame({'value': []})
    if cur == 'USD':
        return _yf_download('USDKRW=X', start, end, diag)
    a = _yf_download(f'{cur}USD=X', start, end, diag)
    b = _yf_download('USDKRW=X', start, end, diag)
    if a.empty or b.empty:
        return pd.DataFrame({'value': []})
    df = a.join(b, how='inner', lsuffix='_a', rsuffix='_b')
    if df.empty:
        return pd.DataFrame({'value': []})
    df['value'] = df['value_a'] * df['value_b']
    df = df[['value']]
    return df

def fetch_series(currency: str, start_date: date, end_date: date, diag) -> pd.DataFrame:
    df = _yf_download(_yf_pair_symbol(currency), start_date, end_date, diag)
    if df.empty:
        df = _yf_cross(currency, start_date, end_date, diag)
    if df.empty:
        s, e = _yyyy_mm_dd(start_date), _yyyy_mm_dd(end_date)
        df = _host_timeseries_direct(currency, s, e, diag)
    if df.empty:
        df = _host_timeseries_inverse(currency, s, e, diag)
    if not df.empty:
        rng = pd.date_range(start=start_date, end=end_date, freq='D')
        df = df.reindex(rng).ffill()
        df.index.name = 'date'
        df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
    return df

def last_available_rate(df: pd.DataFrame, end_dt: pd.Timestamp) -> tuple[pd.Timestamp, float]:
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
        # 우선 xlsxwriter 시도 (빠르고 안정적)
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    except Exception:
        # xlsxwriter 미설치 등 예외 시 openpyxl로 재시도
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    out.seek(0)
    return out.read()

st.title(APP_TITLE)
st.caption('기본 소스: Yahoo Finance (yfinance) → 실패 시 exchangerate.host. 실제 거래/송금 환율과 다를 수 있습니다.')

with st.sidebar:
    st.header('설정 / 입력')
    cur = st.selectbox('통화 선택', options=list(CURRENCIES.keys()), format_func=lambda k: f"{CURRENCIES[k]}")
    today = date.today()
    start_default = today - timedelta(days=90)
    start_dt = st.date_input('시작일', start_default, max_value=today)
    end_dt = st.date_input('종료일', today, min_value=start_dt, max_value=today)
    show_diag = st.checkbox('진단(요청/응답 로그) 보기', value=True)
    run = st.button('그래프 그리기', type='primary')

if run:
    diag = []
    if not YF_AVAILABLE:
        diag.append('yfinance가 설치되어 있지 않습니다. 먼저 설치하세요: pip install yfinance')

    data = fetch_series(cur, start_dt, end_dt, diag)

    if show_diag:
        with st.expander('진단(요청 URL 및 상태)'):
            for line in diag:
                st.code(line)

    if data.empty:
        st.warning('표시할 데이터가 없습니다. 기간을 넓혀 보시거나, 다른 통화를 선택해 보세요. (yfinance 설치를 권장)')
        st.stop()

    import pandas as pd
    end_ts = pd.to_datetime(end_dt.isoformat())
    last_dt, rate = last_available_rate(data, end_ts)
    banner_html = (
        '<div style="padding:12px 16px;border-radius:12px;background:#f5f5f5;font-size:18px;">'
        f'<b>최종 환율</b> — {cur}: {rate:,.4f} KRW (기준일 {last_dt.date().isoformat()})'
        '</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    st.subheader(f'일별 환율 추이 (KRW per 1 {cur})')
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(data.index, data['value'], label=f'{cur}/KRW')
    ax.set_xlabel('날짜')
    ax.set_ylabel(f'KRW per 1 {cur}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.subheader('데이터 표')
    st.dataframe(data.rename(columns={'value': f'{cur}/KRW'}).round(6))
    st.download_button(
        'CSV 다운로드',
        data=to_csv_bytes(data.rename(columns={'value': f'{cur}/KRW'})),
        file_name=f'krw_fx_{cur}_{start_dt.isoformat()}_{end_dt.isoformat()}.csv',
        mime='text/csv',
    )

    st.download_button(
        '엑셀(.xlsx) 다운로드',
        data=to_excel_bytes(data.rename(columns={'value': f'{cur}/KRW'})),
        file_name=f'krw_fx_{cur}_{start_dt.isoformat()}_{end_dt.isoformat()}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

st.markdown('---')
st.caption('ⓘ 주말/공휴일 공시 공백을 대비하여 이전 값(FFill)으로 자동 보정합니다.')
