import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import requests

# --- ページ設定 ---
st.set_page_config(page_title="東京 気温予測AI", page_icon="🌤️", layout="wide")

st.title("🌤️ 名古屋 気温予測AIアプリ (最高・最低気温 対応版)")
st.write("過去30年分のデータを学習したAIが、「平均・最高・最低」の3つの気温を同時に予測します。")

# --- データの読み込みと特徴量作成 ---
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("tokyo_weather_30years_final.csv")
    
    # 3つの気温データをすべてクリーニング
    for col in ["平均気温(℃)", "最高気温(℃)", "最低気温(℃)"]:
        df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=["平均気温(℃)", "最高気温(℃)", "最低気温(℃)", "年", "月", "日"])
    
    df["天気概況(昼)"] = df["天気概況(昼)"].fillna("")
    df["天気_晴れフラグ"] = df["天気概況(昼)"].str.contains("晴").astype(int)
    df["天気_雨雪フラグ"] = df["天気概況(昼)"].str.contains("雨|雪").astype(int)
    
    df_dates = df[['年', '月', '日']].rename(columns={'年': 'year', '月': 'month', '日': 'day'})
    df['日付'] = pd.to_datetime(df_dates, errors='coerce')
    df = df.dropna(subset=['日付']).sort_values('日付').reset_index(drop=True)
    
    # 特徴量（ヒント）に最高・最低気温も追加
    df['1日前の平均気温'] = df['平均気温(℃)'].shift(1)
    df['1日前の最高気温'] = df['最高気温(℃)'].shift(1)
    df['1日前の最低気温'] = df['最低気温(℃)'].shift(1)
    df['2日前の平均気温'] = df['平均気温(℃)'].shift(2)
    df['過去7日間平均'] = df['1日前の平均気温'].rolling(window=7).mean()
    
    df = df.dropna().reset_index(drop=True)
    return df

# --- AIモデルの学習 ---
@st.cache_resource
def train_hybrid_model(data):
    features = [
        '1日前の平均気温', '1日前の最高気温', '1日前の最低気温', 
        '2日前の平均気温', '過去7日間平均', '月', 
        '天気_晴れフラグ', '天気_雨雪フラグ'
    ]
    X = data[features]
    # ⭐️ 予測したい正解データ（ターゲット）を3つに設定
    y = data[['平均気温(℃)', '最高気温(℃)', '最低気温(℃)']]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, features

# 実行処理
df = load_and_prep_data()
model, feature_names = train_hybrid_model(df)

# ==========================================
# 🌍 明日のリアルタイム予測
# ==========================================
st.markdown("---")
st.header("🌍 明日の気温をリアルタイム予測")
st.write("最新の気象データをもとに、明日の平均・最高・最低気温を予測します。")

if st.button("🔄 最新のデータを取得して明日を予測する", type="primary"):
    with st.spinner("インターネットから最新の気象データを取得中..."):
        try:
            # Open-Meteo APIに、最高・最低気温の取得リクエストも追加
            url = "https://api.open-meteo.com/v1/forecast?latitude=35.6895&longitude=139.6917&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,weathercode&past_days=7&forecast_days=2&timezone=Asia%2FTokyo"
            response = requests.get(url)
            api_data = response.json()
            
            daily = api_data['daily']
            dates = daily['time']
            temps_mean = daily['temperature_2m_mean']
            temps_max = daily['temperature_2m_max']
            temps_min = daily['temperature_2m_min']
            codes = daily['weathercode']
            
            today_mean = temps_mean[7]
            today_max = temps_max[7]
            today_min = temps_min[7]
            
            yesterday_mean = temps_mean[6]
            
            past_7_avg = sum(temps_mean[1:8]) / 7.0 
            
            tomorrow_date_str = dates[8]
            tomorrow_month = datetime.strptime(tomorrow_date_str, "%Y-%m-%d").month
            
            tomorrow_code = codes[8]
            is_sunny = 1 if tomorrow_code <= 2 else 0
            is_rain_snow = 1 if tomorrow_code >= 51 else 0
            
            X_real = pd.DataFrame([{
                '1日前の平均気温': today_mean,
                '1日前の最高気温': today_max,
                '1日前の最低気温': today_min,
                '2日前の平均気温': yesterday_mean,
                '過去7日間平均': past_7_avg,
                '月': tomorrow_month,
                '天気_晴れフラグ': is_sunny,
                '天気_雨雪フラグ': is_rain_snow
            }])
            
            # AIで3つの気温を同時に予測
            predictions = model.predict(X_real)[0]
            pred_avg, pred_max, pred_min = predictions
            
            st.success("データの取得と予測が完了しました！")
            st.subheader(f"📅 明日 ({tomorrow_date_str}) の予測")
            
            weather_text = "晴れ/曇り系 🌤️" if is_sunny else ("雨/雪系 ☔" if is_rain_snow else "その他")
            st.write(f"**📡 明日の予報天気:** {weather_text}")
            
            # ⭐️ 画面に平均・最高・最低気温を表示する部分
            col1, col2, col3 = st.columns(3)
            col1.metric("🤖 平均気温", f"{pred_avg:.1f} ℃")
            col2.metric("🔺 最高気温", f"{pred_max:.1f} ℃")
            col3.metric("🔻 最低気温", f"{pred_min:.1f} ℃")
            
            with st.expander("APIから取得した『AIへのヒント（入力データ）』を確認"):
                st.dataframe(X_real)
                
        except Exception as e:
            st.error(f"データの取得に失敗しました。（エラー詳細: {e}）")

# ==========================================
# 🔍 過去のデータで検証
# ==========================================
st.markdown("---")
st.header("🔍 過去のデータでAIの精度を検証する")

with st.expander("過去の予測精度と特徴量重要度を見る（クリックで展開）"):
    min_date = df['日付'].max() - timedelta(days=365)
    max_date = df['日付'].max()

    selected_date = st.date_input(
        "予測テストを行う日付を選択",
        value=max_date - timedelta(days=1),
        min_value=min_date,
        max_value=max_date
    )

    target_data = df[df['日付'] == pd.to_datetime(selected_date)]

    if not target_data.empty:
        X_input = target_data[feature_names]
        
        # 過去データでも3つ同時に予測
        preds = model.predict(X_input)[0]
        pred_avg, pred_max, pred_min = preds
        
        actual_avg = target_data['平均気温(℃)'].values[0]
        actual_max = target_data['最高気温(℃)'].values[0]
        actual_min = target_data['最低気温(℃)'].values[0]
        actual_weather = target_data['天気概況(昼)'].values[0]
        
        st.write(f"**この日の実際の天気:** {actual_weather}")
        
        # ⭐️ 画面に平均・最高・最低気温を表示する部分
        st.markdown("##### 🌡️ 平均気温")
        c1, c2, c3 = st.columns(3)
        c1.metric("🤖 AI予測", f"{pred_avg:.1f} ℃")
        c2.metric("🎯 実際", f"{actual_avg:.1f} ℃")
        c3.metric("📉 誤差", f"{abs(pred_avg - actual_avg):.1f} ℃", delta_color="inverse")
        
        st.markdown("##### 🔺 最高気温")
        c4, c5, c6 = st.columns(3)
        c4.metric("🤖 AI予測", f"{pred_max:.1f} ℃")
        c5.metric("🎯 実際", f"{actual_max:.1f} ℃")
        c6.metric("📉 誤差", f"{abs(pred_max - actual_max):.1f} ℃", delta_color="inverse")
        
        st.markdown("##### 🔻 最低気温")
        c7, c8, c9 = st.columns(3)
        c7.metric("🤖 AI予測", f"{pred_min:.1f} ℃")
        c8.metric("🎯 実際", f"{actual_min:.1f} ℃")
        c9.metric("📉 誤差", f"{abs(pred_min - actual_min):.1f} ℃", delta_color="inverse")
        
        st.subheader("🧠 特徴量重要度（予測の根拠）")
        importance_df = pd.DataFrame({"特徴量": feature_names, "重要度": model.feature_importances_})
        st.bar_chart(importance_df.sort_values(by="重要度", ascending=False).set_index("特徴量"))
