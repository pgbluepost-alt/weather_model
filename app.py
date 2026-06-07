import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- ページ設定 ---
st.set_page_config(page_title="東京 気温予測AI", page_icon="🌤️", layout="wide")

st.title("🌤️ 東京 気温予測AIアプリ")
st.write("過去30年分の気象データ（気象庁）を学習したAIが、指定した日付の気温を予測します。")

# --- データの読み込みとクリーニング ---
@st.cache_data
def load_data():
    df = pd.read_csv("tokyo_weather_30years_final.csv")
    
    # 【追加処理】気温データから `)` や `]` などの記号を取り除き、数値に変換する
    df["平均気温(℃)"] = df["平均気温(℃)"].astype(str).str.replace(r'[^\d.-]', '', regex=True)
    df["平均気温(℃)"] = pd.to_numeric(df["平均気温(℃)"], errors='coerce')
    
    # 欠損値を削除
    df = df.dropna(subset=["平均気温(℃)", "月", "日"])
    return df

# --- AIモデルの学習 ---
@st.cache_resource
def train_model(data):
    X = data[["月", "日"]]
    y = data["平均気温(℃)"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

df = load_data()
model = train_model(df)

# --- サイドバー ---
st.sidebar.header("📅 予測したい日付を選択")
target_month = st.sidebar.slider("月 (Month)", 1, 12, 6)
target_day = st.sidebar.slider("日 (Day)", 1, 31, 15)

st.markdown("---")

# --- 予測と表示 ---
input_data = pd.DataFrame({"月": [target_month], "日": [target_day]})
predicted_temp = model.predict(input_data)[0]

st.header(f"✨ 予測結果：{target_month}月{target_day}日")
st.metric(label="AI予測 平均気温", value=f"{predicted_temp:.1f} ℃")

st.subheader(f"📊 過去30年間の {target_month}月{target_day}日 の気温推移")
past_data = df[(df["月"] == target_month) & (df["日"] == target_day)].copy()

if not past_data.empty:
    chart_data = past_data.set_index("年")[["平均気温(℃)", "最高気温(℃)", "最低気温(℃)"]]
    
    # グラフ用にもクリーニング（念のため最高・最低気温の記号も除去）
    for col in chart_data.columns:
        chart_data[col] = chart_data[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        chart_data[col] = pd.to_numeric(chart_data[col], errors='coerce')
        
    st.line_chart(chart_data)
else:
    st.warning("この日付の過去データが見つかりませんでした。")
