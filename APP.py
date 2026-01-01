import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

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
    # System 1
    dist_thres = st.number_input("Distance Threshold (Chop Zone)", value=0.5, step=0.1, help="ระยะห่างจากเส้น 0 เพื่อเลี่ยงไซด์เวย์")
    
    # System 2
    show_div = st.checkbox("Show Divergence", value=True)
    lb_r = st.number_input("Pivot Lookback Right", value=5, min_value=1)
    lb_l = st.number_input("Pivot Lookback Left", value=5, min_value=1)

    # System 3
    show_dash = st.checkbox("Show HTF Dashboard", value=True)
    htf_res = st.selectbox("Higher Timeframe (Dashboard)", ["1wk", "1mo", "3mo"], index=0)

# ==========================================
# 2. FUNCTIONS (ฟังก์ชันคำนวณ)
# ==========================================
def calculate_macd(df, fast, slow, signal):
    # Standard MACD Formula
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist_line = macd_line - signal_line
    return macd_line, signal_line, hist_line

def detect_divergence(df, lb_l, lb_r):
    # Simple Pivot Point Logic matching Pine Script ta.pivotlow/high
    # Note: This is a simplified retrospective check
    df['div_bull'] = np.nan
    df['div_bear'] = np.nan
    
    # Iterate to find pivots (Need lookahead for right side, so we simulate)
    # Pivot Low: Lowest in window [i-lb_l : i+lb_r]
    # Pivot High: Highest in window [i-lb_l : i+lb_r]
    
    pivots_low = []
    pivots_high = []

    # Store (index, value)
    last_pl_macd = None
    last_pl_price = None
    last_ph_macd = None
    last_ph_price = None

    for i in range(lb_l, len(df) - lb_r):
        # Check Pivot Low (MACD)
        window = df['MACD'].iloc[i-lb_l : i+lb_r+1]
        if df['MACD'].iloc[i] == window.min():
            # Found Pivot Low
            current_pl_macd = df['MACD'].iloc[i]
            current_pl_price = df['Low'].iloc[i]
            
            # Check Bullish Divergence
            # Price Lower Low AND MACD Higher Low
            if last_pl_macd is not None:
                if current_pl_price < last_pl_price and current_pl_macd > last_pl_macd:
                    df.at[df.index[i], 'div_bull'] = current_pl_macd # Mark on MACD

            last_pl_macd = current_pl_macd
            last_pl_price = current_pl_price

        # Check Pivot High (MACD)
        if df['MACD'].iloc[i] == window.max():
            # Found Pivot High
            current_ph_macd = df['MACD'].iloc[i]
            current_ph_price = df['High'].iloc[i]
            
            # Check Bearish Divergence
            # Price Higher High AND MACD Lower High
            if last_ph_macd is not None:
                if current_ph_price > last_ph_price and current_ph_macd < last_ph_macd:
                    df.at[df.index[i], 'div_bear'] = current_ph_macd # Mark on MACD
            
            last_ph_macd = current_ph_macd
            last_ph_price = current_ph_price
            
    return df

# ==========================================
# 3. MAIN LOGIC
# ==========================================

# 3.1 Fetch Data (Main Timeframe)
data = yf.download(ticker, period=period, interval=timeframe, progress=False)

if len(data) > 0:
    # คำนวณ MACD หลัก
    data['MACD'], data['Signal'], data['Hist'] = calculate_macd(data, fast_len, slow_len, sig_len)
    
    # 3.2 Determine Histogram Colors (For visualization)
    # Logic: hist > 0 ? (rising ? green : lightgreen) ...
    data['Hist_Color'] = np.where(data['Hist'] >= 0, 
                                  np.where(data['Hist'] > data['Hist'].shift(1), 'rgba(0, 230, 118, 0.9)', 'rgba(0, 230, 118, 0.4)'),
                                  np.where(data['Hist'] < data['Hist'].shift(1), 'rgba(255, 82, 82, 0.9)', 'rgba(255, 82, 82, 0.4)'))

    # 3.3 System 1: MACD Line Color logic based on Chop Zone
    # Note: Plotly lines have single color, we will use Scatter markers or segments to simulate multicolor line
    # Or simply plot regions. Here we categorize for the dashboard/logic.
    conditions = [
        (data['MACD'] > dist_thres),
        (data['MACD'] < -dist_thres)
    ]
    choices = ['Bullish Zone', 'Bearish Zone']
    data['Zone'] = np.select(conditions, choices, default='Chop Zone')

    # 3.4 System 2: Divergence
    if show_div:
        data = detect_divergence(data, lb_l, lb_r)

    # 3.5 System 3: HTF Dashboard Data
    htf_bias = "N/A"
    htf_color = "gray"
    
    if show_dash:
        try:
            htf_data = yf.download(ticker, period="1y", interval=htf_res, progress=False)
            if len(htf_data) > 0:
                m, s, h = calculate_macd(htf_data, fast_len, slow_len, sig_len)
                current_htf_macd = m.iloc[-1]
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
    
    # Create Subplots (Row 1: Price, Row 2: MACD)
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

    # B. Chop Zone Fill (Gray Area)
    # We use a shape to represent the chop zone across the entire chart
    fig.add_hrect(y0=-dist_thres, y1=dist_thres, 
                  fillcolor="gray", opacity=0.15, line_width=0, row=2, col=1,
                  annotation_text="Chop Zone", annotation_position="left")
    
    # C. Chop Limit Lines
    fig.add_hline(y=dist_thres, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=-dist_thres, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=0, line_color="white", row=2, col=1)

    # D. MACD Line (Multi-color trick)
    # Plotly doesn't support multi-color lines easily. We will plot points or overlay lines.
    # Approach: Plot gray line as base, then overlay Green/Red parts.
    
    # Base Line (Gray - Chop)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                             line=dict(color='gray', width=1), name='MACD (Chop)'), row=2, col=1)
    
    # Bullish Line (Green > Thres)
    bull_macd = data['MACD'].copy()
    bull_macd[bull_macd <= dist_thres] = np.nan
    fig.add_trace(go.Scatter(x=data.index, y=bull_macd, 
                             line=dict(color='#00E676', width=2), name='MACD (Bull)'), row=2, col=1)

    # Bearish Line (Red < -Thres)
    bear_macd = data['MACD'].copy()
    bear_macd[bear_macd >= -dist_thres] = np.nan
    fig.add_trace(go.Scatter(x=data.index, y=bear_macd, 
                             line=dict(color='#FF5252', width=2), name='MACD (Bear)'), row=2, col=1)

    # Signal Line
    fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], 
                             line=dict(color='orange', width=1), name='Signal'), row=2, col=1)

    # E. Divergence Markers
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

    # Layout Updates
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
