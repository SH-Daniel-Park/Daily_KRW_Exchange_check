# KRW FX Chart (ECOS)

ECOS 731Y001(주요국 통화의 대원화환율, 주기 D) 기반 Streamlit 앱.

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 배포(Streamlit Cloud)
1) 이 폴더를 공개 GitHub 저장소에 올림
2) share.streamlit.io → New app → 파일: `streamlit_app.py`
3) 앱 대시보드 → ⋯ → Edit secrets → 아래처럼 추가
```
ECOS_API_KEY = "여기에_본인_키"
```
