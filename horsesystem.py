import streamlit as st
import pandas as pd
import boto3
import os
from decimal import Decimal
from google import genai
import plotly.express as px
import plotly.graph_objects as go

# --- ページ設定 ---
st.set_page_config(page_title="🐎 軽種馬育成分析", layout="wide")
st.title("🐎 軽種馬育成分析ダッシュボード")
st.markdown("DynamoDB（HorseTrainingData）から実データを取得し、トレーニングの質とフォームを分析します。")

# --- 環境変数と設定 ---
TABLE_NAME = os.environ.get("TABLE_NAME", "HorseTrainingData")

# --- 1. DynamoDBからのデータ取得 ---
@st.cache_data(ttl=60)
def load_production_data():
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(TABLE_NAME)
        
        response = table.scan(Limit=1000) 
        items = response.get('Items', [])
        
        if not items:
            return pd.DataFrame()
            
        df = pd.DataFrame(items)
        
        # DynamoDBのDecimal型をfloatに変換
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, Decimal)).any():
                df[col] = df[col].astype(float)
                
        # timestamp(エポックms)でソート
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        # 歩様(Gait)を人間が読みやすいラベルに変換
        gait_map = {0: "0:停止", 1: "1:常歩(Walk)", 2: "2:速歩(Trot)", 3: "3:駈歩(Canter)"}
        if "gait" in df.columns:
            df["gait_label"] = df["gait"].map(gait_map)
            
        return df
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return pd.DataFrame()

df = load_production_data()

if df.empty:
    st.warning("データが存在しません。センサーデバイスからのデータ送信を確認してください。")
    st.stop()

# --- 2. データの表（テーブル）を表示 ---
st.subheader("📋 最新のトレーニングデータ")
display_cols = ["time", "gait_label", "rhythm_pitch", "rhythm_stability", "balance_lr", "body_wobble", "energy_consumption"]
available_cols = [c for c in display_cols if c in df.columns]
st.dataframe(df[available_cols], use_container_width=True)

# --- 3. グラフの表示 (Plotlyによる可視化強化) ---
st.subheader("📈 トレーニング指標の推移")

# 時間軸インデックス化
if "time" in df.columns:
    df["time_str"] = df["time"].astype(str)

# 6つのタブを作成
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "歩様の推移", 
    "ピッチ", 
    "リズム・ピッチの安定性", 
    "左右のバランス", 
    "体幹のぶれ", 
    "1秒あたりのエネルギー"
])

with tab1:
    st.markdown("**時間経過に伴う歩様（運動強度）の変化**")
    if "gait_label" in df.columns:
        fig_gait = px.scatter(
            df, x="time_str", y="gait_label", color="gait_label",
            category_orders={"gait_label": ["3:駈歩(Canter)", "2:速歩(Trot)", "1:常歩(Walk)", "0:停止"]},
            labels={"time_str": "時間", "gait_label": "歩様"},
            title="トレーニングメニューの推移（ステップグラフ）"
        )
        fig_gait.update_traces(mode='lines+markers', line_shape='hv')
        st.plotly_chart(fig_gait, use_container_width=True)

with tab2:
    st.markdown("**ピッチ（歩行のテンポ）の推移**")
    if "rhythm_pitch" in df.columns:
        fig_pitch = px.line(df, x="time_str", y="rhythm_pitch", labels={"time_str": "時間", "rhythm_pitch": "ピッチ (Hz)"}, title="ピッチの時系列変化")
        st.plotly_chart(fig_pitch, use_container_width=True)
    else:
        st.info("データがありません。")

with tab3:
    st.markdown("**リズム・ピッチの安定性 (数値が低いほど安定)**")
    if "rhythm_stability" in df.columns:
        fig_stability = px.line(df, x="time_str", y="rhythm_stability", labels={"time_str": "時間", "rhythm_stability": "安定性 (標準偏差)"}, title="リズム・ピッチの安定性")
        st.plotly_chart(fig_stability, use_container_width=True)
    else:
        st.info("データがありません。")

with tab4:
    st.markdown("**左右のバランス**")
    if "balance_lr" in df.columns:
        fig_balance = px.line(df, x="time_str", y="balance_lr", labels={"time_str": "時間", "balance_lr": "左右バランス"}, title="左右のバランス推移")
        st.plotly_chart(fig_balance, use_container_width=True)
    else:
        st.info("データがありません。")

with tab5:
    st.markdown("**体幹のぶれ (deg/s)**")
    if "body_wobble" in df.columns:
        fig_wobble = px.line(df, x="time_str", y="body_wobble", labels={"time_str": "時間", "body_wobble": "体幹のブレ (deg/s)"}, title="体幹のぶれ推移")
        fig_wobble.update_traces(line=dict(color='red'))
        st.plotly_chart(fig_wobble, use_container_width=True)
    else:
        st.info("データがありません。")

with tab6:
    st.markdown("**1秒あたりのエネルギー消費**")
    if "energy_consumption" in df.columns:
        fig_energy = px.line(df, x="time_str", y="energy_consumption", labels={"time_str": "時間", "energy_consumption": "エネルギー消費"}, title="1秒あたりのエネルギー推移")
        fig_energy.update_traces(line=dict(color='orange'), fill='tozeroy')
        st.plotly_chart(fig_energy, use_container_width=True)
    else:
        st.info("データがありません。")


# --- 4. AI分析メニュー (LLM連携の準備) ---
st.subheader("🤖 AI分析・コーチング")
analysis_choice = st.selectbox(
    "どの観点でAIに分析させますか？",
    [
        "本日のトレーニング強度とエネルギー消費の総括",
        "各歩様（常歩・速歩・駈歩）におけるピッチの安定性とバランス評価",
        "体幹のブレ（body_wobble）やリズムの乱れに基づくフォームの崩れチェック"
    ]
)

# --- 5. 本番用AI分析の実行 ---
if st.button("AI分析を実行する"):
    with st.spinner("初期調教専門家及び獣医師AIがデータを分析中..."):
        stat_cols = ["rhythm_pitch", "rhythm_stability", "balance_lr", "body_wobble", "energy_consumption"]
        exist_cols = [c for c in stat_cols if c in df.columns]
        
        if "gait_label" in df.columns:
            summary_stats = df.groupby("gait_label")[exist_cols].mean().to_string()
        else:
            summary_stats = df[exist_cols].describe().to_string()
        
        prompt = f"""
        あなたは競走馬初期調教の専門家および競走馬の獣医師です。
        以下のセンサーデータ統計値（歩様ごとの平均値）をもとに、「{analysis_choice}」という観点で専門的な分析レポートを作成してください。

        【データの前提知識】
        - rhythm_pitch: 歩行のテンポ。
        - rhythm_stability: 上下動の標準偏差。数値が**小さいほどリズムが安定**している。
        - balance_lr: 左右のバランス。0に近いほど均等。
        - body_wobble: 体幹のブレ（ジャイロRMS）。数値が**大きいほどフォームが崩れている**。
        - energy_consumption: 単位時間あたりの運動エネルギー。

        【データ統計値】
        {summary_stats}

        【出力形式】
        ・重要な気づきを箇条書きで3点。
        ・トレーナー・騎手に対する具体的な改善アドバイスを1段落。
        """
        
        try:
            client =  genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=prompt,
            )
            result_text = response.text
            
            st.success("分析が完了しました。")
            st.info(result_text)
            
        except Exception as e:
            st.error(f"AI分析中にエラーが発生しました: {e}")

# --- 6. CSVダウンロードボタン ---
st.subheader("📥 CSVエクスポート")
csv_data = df.to_csv(index=False).encode('utf-8-sig') 
st.download_button(
    label="全データをCSVとしてダウンロード",
    data=csv_data,
    file_name="production_horse_data.csv",
    mime="text/csv"
)
