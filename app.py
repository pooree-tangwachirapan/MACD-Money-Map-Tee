import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="MACD Money Map", layout="wide")
st.title("🗺️ MACD Money Map [Python Version]")

# ==========================================
# 1. SIDEBAR SETTINGS (การตั้งค่า)
# ==========================================
with st.sidebar:
    st.header("⚙️ Data Settings")
    ticker = st.text_input("Symbol (e.g., BTC-USD, AAPL)", "BTC-USD")
    timeframe = st.selectbox("Timeframe", ["1d", "1wk", "1mo"], index=0)
    period = st.selectbox("Data Period", ["1y", "2y", "5y", "max"], index=0)

    st.header("📊 MACD Settings")
    fast_len = st.number_input("Fast Length", value=12)
    slow_len = st.number_input("Slow Length", value=26)
    sig_len = st.number_input("Signal Smoothing", value=9)

    st.header("💰 Money Map Rules")
    # System 1: Distance Rule
    dist_thres = st.number_input("Distance Threshold (Chop Zone)", value=0.5, step=0.1, help="ระยะห่างจากเส้น 0 เพื่อเลี่ยงไซด์เวย์")
    
    # System 2: Divergence
    show_div = st.checkbox("Show Divergence", value=True)
    lb_r = st.number_input("Pivot Lookback Right", value=5, min_value=1)
    lb_l = st.number_input("Pivot Lookback Left", value=5, min_value=1)

    # System 3: Dashboard
    show_dash = st.checkbox("Show HTF Dashboard", value=True)
    htf_res = st.selectbox("Higher Timeframe (Dashboard)", ["1wk", "1mo", "3mo"], index=0)

# ==========================================
# 2. FUNCTIONS (ฟังก์ชันคำนวณ)
# ==========================================
def flatten_data(df):
    """ฟังก์ชันช่วยแปลง MultiIndex จาก yfinance ให้เป็น Single Index"""
    # ถ้า Column เป็น MultiIndex (มีชื่อหุ้นห้อยท้าย) ให้เอาชื่อหุ้นออก
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_macd(df, fast, slow, signal):
    # Standard MACD Formula
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist_line = macd_line - signal_line
    return macd_line, signal_line, hist_line

def detect_divergence(df, lb_l, lb_r):
    # เตรียมคอลัมน์
    df['div_bull'] = np.nan
    df['div_bear'] = np.nan
    
    # ตัวแปรเก็บค่าล่าสุดที่เจอ Pivot
    last_pl_macd = None
    last_pl_price = None
    last_ph_macd = None
    last_ph_price = None

    # วนลูปตรวจสอบ Pivot (ต้องเว้นระยะ lb_l และ lb_r)
    # ใช้ float() ครอบเพื่อป้องกัน Error: The truth value of a Series is ambiguous
    for i in range(lb_l, len(df) - lb_r):
        
        # --- 1. ตรวจสอบ Pivot Low (MACD) ---
        window_macd = df['MACD'].iloc[i-lb_l : i+lb_r+1]
        
        if df['MACD'].iloc[i] == window_macd.min():
            # บังคับเป็น float ทันทีเพื่อแก้ปัญหาข้อมูลที่เป็น Series
            current_pl_macd = float(df['MACD'].iloc[i]) 
            current_pl_price = float(df['Low'].iloc[i]) 
            
            # Logic Bullish Divergence
            if last_pl_macd is not None:
                if current_pl_price < last_pl_price and current_pl_macd > last_pl_macd:
                    df.at[df.index[i], 'div_bull'] = current_pl_macd

            last_pl_macd = current_pl_macd
            last_pl_price = current_pl_price

        # --- 2. ตรวจสอบ Pivot High (MACD) ---
        if df['MACD'].iloc[i] == window_macd.max():
            current_ph_macd = float(df['MACD'].iloc[i])
            current_ph_price = float(df['High'].iloc[i])
            
            # Logic Bearish Divergence
            if last_ph_macd is not None:
                if current_ph_price > last_ph_price and current_ph_macd < last_ph_macd:
                    df.at[df.index[i], 'div_bear'] = current_ph_macd
            
            last_ph_macd = current_ph_macd
            last_ph_price = current_ph_price
            
    return df

# ==========================================
# 3. MAIN LOGIC
# ==========================================

# 3.1 Fetch Data
try:
    data = yf.download(ticker, period=period, interval=timeframe, progress=False)
    
    # --- FIX BUG: จัดการข้อมูล yfinance ที่เป็น MultiIndex ---
    data = flatten_data(data)
    
    if len(data) > 0:
        # คำนวณ MACD หลัก
        data['MACD'], data['Signal'], data['Hist'] = calculate_macd(data, fast_len, slow_len, sig_len)
        
        # 3.2 กำหนดสี Histogram
        hist_diff = data['Hist'].diff()
        data['Hist_Color'] = np.where(data['Hist'] >= 0, 
                                      np.where(hist_diff > 0, 'rgba(0, 230, 118, 0.9)', 'rgba(0, 230, 118, 0.4)'),
                                      np.where(hist_diff < 0, 'rgba(255, 82, 82, 0.9)', 'rgba(255, 82, 82, 0.4)'))

        # 3.3 คำนวณ Divergence (ถ้าเปิดใช้งาน)
        if show_div:
            data = detect_divergence(data, lb_l, lb_r)

        # 3.4 HTF Dashboard Data
        htf_bias = "N/A"
        htf_color = "gray"
        
        if show_dash:
            try:
                htf_data = yf.download(ticker, period="1y", interval=htf_res, progress=False)
                htf_data = flatten_data(htf_data) # Flatten ข้อมูล HTF ด้วย
                
                if len(htf_data) > 0:
                    m, s, h = calculate_macd(htf_data, fast_len, slow_len, sig_len)
                    current_htf_macd = float(m.iloc[-1])
                    if current_htf_macd > 0:
                        htf_bias = "BULLISH (Only Buy)"
                        htf_color = "green"
                    else:
                        htf_bias = "BEARISH (Only Sell)"
                        htf_color = "red"
            except Exception as e:
                htf_bias = f"Error: {e}"

        # ==========================================
        # 4. VISUALIZATION (Plotly)
        # ==========================================
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.5, 0.5])

        # --- Chart 1: Price ---
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'], high=data['High'],
                                     low=data['Low'], close=data['Close'],
                                     name='Price'), row=1, col=1)

        # --- Chart 2: MACD Money Map ---
        
        # A. Histogram
        fig.add_trace(go.Bar(x=data.index, y=data['Hist'], 
                             marker_color=data['Hist_Color'], name='Histogram'), row=2, col=1)

        # B. Chop Zone
        fig.add_hrect(y0=-dist_thres, y1=dist_thres, 
                      fillcolor="gray", opacity=0.15, line_width=0, row=2, col=1,
                      annotation_text="Chop Zone", annotation_position="top left")
        
        fig.add_hline(y=dist_thres, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=-dist_thres, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=0, line_color="white", opacity=0.3, row=2, col=1)

        # C. MACD Line (แยกสีตามโซน)
        # 1. Base Line (Chop Zone - สีเทา)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                                 line=dict(color='gray', width=1), name='MACD (Chop)'), row=2, col=1)
        
        # 2. Bullish Line (Green > Threshold)
        bull_macd = data['MACD'].copy()
        bull_macd[bull_macd <= dist_thres] = np.nan
        fig.add_trace(go.Scatter(x=data.index, y=bull_macd, 
                                 line=dict(color='#00E676', width=2), name='MACD (Bull)'), row=2, col=1)

        # 3. Bearish Line (Red < -Threshold)
        bear_macd = data['MACD'].copy()
        bear_macd[bear_macd >= -dist_thres] = np.nan
        fig.add_trace(go.Scatter(x=data.index, y=bear_macd, 
                                 line=dict(color='#FF5252', width=2), name='MACD (Bear)'), row=2, col=1)

        # Signal Line
        fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], 
                                 line=dict(color='orange', width=1), name='Signal'), row=2, col=1)

        # D. Divergence Markers
        if show_div:
            # Bull Div
            bull_div_data = data.dropna(subset=['div_bull'])
            if not bull_div_data.empty:
                fig.add_trace(go.Scatter(x=bull_div_data.index, y=bull_div_data['div_bull'],
                                         mode='markers+text', marker_symbol='triangle-up', 
                                         marker_color='green', marker_size=10,
                                         text="Bull Div", textposition="bottom center",
                                         name='Bull Div'), row=2, col=1)
                
            # Bear Div
            bear_div_data = data.dropna(subset=['div_bear'])
            if not bear_div_data.empty:
                fig.add_trace(go.Scatter(x=bear_div_data.index, y=bear_div_data['div_bear'],
                                         mode='markers+text', marker_symbol='triangle-down', 
                                         marker_color='red', marker_size=10,
                                         text="Bear Div", textposition="top center",
                                         name='Bear Div'), row=2, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                          template="plotly_dark", title_text=f"{ticker} Analysis")
        
        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # 5. DASHBOARD DISPLAY
        # ==========================================
        if show_dash:
            st.markdown("### 📊 Higher Timeframe Dashboard")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("HTF Timeframe", htf_res)
            with col2:
                st.metric("Current Trend", htf_bias)
            with col3:
                status_color = "🟢" if htf_color == "green" else "🔴"
                st.write(f"## {status_color}")
            
            if htf_color == "green":
                st.success("HTF Bias is BULLISH: Focus on Long setups.")
            elif htf_color == "red":
                st.error("HTF Bias is BEARISH: Focus on Short setups.")

    else:
        st.error(f"No data found for {ticker}. Please check the symbol.")

except Exception as e:
    st.error(f"Critical Error: {e}")
    st.info("Try refreshing the page or checking the stock symbol.")
