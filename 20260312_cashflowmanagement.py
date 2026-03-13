import streamlit as st 
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from urllib.parse import quote
import os
from dotenv import load_dotenv
import time
from streamlit_cookies_controller import CookieController

st.set_page_config(page_title="CASHFLOW MANAGEMENT", layout="wide")

load_dotenv()

# ===================== CACHE FUNCTIONS =====================
@st.cache_data(ttl=10)  # Cache 5 นาที
def read_google_sheet(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    """อ่านข้อมูลจาก Google Sheet พร้อม cache"""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet_name)}"
    df = pd.read_csv(url)
    return df.dropna(how="all")

@st.cache_data(ttl=10)
def preprocess_data(_ap_score, _ar_risk, _ap_bill, _ar_bill):

    _ap_bill['Due'] = pd.to_numeric(_ap_bill['Due'], errors='coerce')
    _ar_bill['Due'] = pd.to_numeric(_ar_bill['Due'], errors='coerce')

    date_cols = ['Date', 'Due date', 'วันที่รับเงินจริง', 'วันที่จ่ายจริง']

    for col in date_cols:
        if col in _ap_bill.columns:
            _ap_bill[col] = pd.to_datetime(
                _ap_bill[col],
                dayfirst=True,
                errors='coerce'
            )

        if col in _ar_bill.columns:
            _ar_bill[col] = pd.to_datetime(
                _ar_bill[col],
                dayfirst=True,
                errors='coerce'
            )

    return _ap_score, _ar_risk, _ap_bill, _ar_bill



@st.cache_data(ttl=10)
def calculate_scores(_ap_score, scenario_weights):
    """คำนวณคะแนนพร้อม cache"""
    df = _ap_score.copy()
    weight_cols = [k for k in scenario_weights.keys() if k in df.columns]
    df['ผลคะแนน'] = sum(scenario_weights[k] * df[k] for k in weight_cols)
    return df

# ===================== โหลดข้อมูล =====================
sheet_id = os.getenv("gglink")

# โหลดข้อมูลพร้อม cache
ap_Score = read_google_sheet(sheet_id, "AP (SCORE)")
ar_Risk = read_google_sheet(sheet_id, "AR (SCORE)")
plan_cashflow_ap = read_google_sheet(sheet_id, "AP(BILL)")
plan_cashflow_ar = read_google_sheet(sheet_id, "AR(BILL)")
if ap_Score.empty or ar_Risk.empty or plan_cashflow_ap.empty or plan_cashflow_ar.empty:
    st.error("❌ ไม่สามารถโหลดข้อมูลได้ กรุณาตรวจสอบ Google Sheet")
    st.stop()
# ประมวลผลข้อมูล
ap_Score, ar_Risk, plan_cashflow_ap, plan_cashflow_ar = preprocess_data(
    ap_Score, ar_Risk, plan_cashflow_ap, plan_cashflow_ar
)
# ===================== SIDEBAR =====================
st.sidebar.header("⚙️ การตั้งค่าหลัก")

เงินสดยกมา = st.sidebar.number_input('💰 CASH', value=0.0, step=10000.0, format="%.0f")
Short_term_loan = st.sidebar.number_input('Short Term Loan', value=0.0, step=10000.0, format="%.0f")
threshold = st.sidebar.number_input('⚠️ Minimum Cash (Threshold)', value=0.0, step=10000.0, format="%.0f")

st.sidebar.markdown("---")

# Weighting Style
st.sidebar.subheader("📊 Weighting Style")
scenario_choice = st.sidebar.radio(
    "เลือกรูปแบบการให้น้ำหนัก:",
    ['Balanced', 'Operational Continuity', 'Liquidity'],
    index=0
)

st.sidebar.markdown("---")

# Defer Style
st.sidebar.subheader("📅 Defer Style")
defer_preset = st.sidebar.radio(
    "เลือกรูปแบบการเลื่อนการจ่าย:",
    ["All item", "Low number of transactions", "Low relationship & low transactions",
     "Low relationship & high transactions", "Custom"],
    index=0
)

# ตั้งค่าตาม preset
preset_configs = {
    "All item": ((1, 5), (1, 5)),
    "Low number of transactions": ((1, 5), (1, 3)),
    "Low relationship & low transactions": ((1, 3), (1, 3)),
    "Low relationship & high transactions": ((1, 3), (3, 5))
}

if defer_preset != "Custom":
    score_range, grade_range = preset_configs[defer_preset]
else:
    st.sidebar.markdown("##### กำหนดเอง:")
    score_range = st.sidebar.slider("Score Range", 1, 5, (1, 5))
    grade_range = st.sidebar.slider("Grade Range", 1, 5, (1, 5))

# ===================== คำนวณคะแนน =====================
scenarios = {
    'Balanced': {'COUNT': 0.1, 'APDAY': 0.1, 'TOTAL': 0.1, 'OPN': 0.2, 'STR': 0.15, 'FLEX': 0.1, 'SUBS': 0.1, 'FIN': 0.15},
    'Operational Continuity': {'COUNT': 0.02, 'APDAY': 0.03, 'TOTAL': 0.1, 'OPN': 0.3, 'STR': 0.15, 'FLEX': 0.05, 'SUBS': 0.2, 'FIN': 0.15},
    'Liquidity': {'COUNT': 0.05, 'APDAY': 0.1, 'TOTAL': 0.05, 'OPN': 0.2, 'STR': 0.05, 'FLEX': 0.2, 'SUBS': 0.1, 'FIN': 0.25}
}

# คำนวณคะแนนพร้อม cache
ap_Score = calculate_scores(ap_Score, scenarios[scenario_choice])

# เตรียมข้อมูล plan_cashflow_ap
plan_cashflow_ap = plan_cashflow_ap.copy()
plan_cashflow_ap['ผลคะแนน'] = plan_cashflow_ap['From'].map(
    dict(zip(ap_Score['รายชื่อเจ้าหนี้'], ap_Score['ผลคะแนน']))
).fillna(1)
plan_cashflow_ap['Ave_ap_day'] = round(
    plan_cashflow_ap['From'].map(dict(zip(ap_Score['รายชื่อเจ้าหนี้'], ap_Score['Average of ap day']))), 0
)
plan_cashflow_ap['credit term'] = (plan_cashflow_ap['Due date'] - plan_cashflow_ap['Date']).dt.days

# แปลงกลับเป็น date
for df in [plan_cashflow_ap, plan_cashflow_ar]:
    df['Date'] = df['Date'].dt.date
    df['Due date'] = df['Due date'].dt.date

# ===================== สร้าง date range =====================
start_date = min(
    plan_cashflow_ap['Due date'].min(),
    plan_cashflow_ar['Due date'].min()
)
end_date = max(
    plan_cashflow_ap['Due date'].max(),
    plan_cashflow_ar['Due date'].max()
)

today = datetime.now().date()
default_start = max(today.replace(day=1), start_date)

selected_dates = st.slider(
    "เลือกช่วงวันที่",
    min_value=start_date,
    max_value=end_date,
    value=(default_start, end_date),
    format="DD/MM/YYYY"
)

filtered_start, filtered_end = selected_dates
date_range = pd.date_range(start=filtered_start, end=filtered_end, freq='D')
df_dates = pd.DataFrame({'Due date': [d.date() for d in date_range]})

# ===================== ฟังก์ชันคำนวณ (Optimized) =====================
def calculate_daily_cashflow(df_dates, plan_ap, plan_ar, opening_cash=0.0, loan=0.0):
    """คำนวณกระแสเงินสดรายวัน (optimized)"""
    # ใช้ groupby แทนการ loop
    ap_daily = plan_ap.groupby('Due date', as_index=False)['Due'].sum().rename(columns={'Due': 'เงินสดจ่าย'})
    ar_daily = plan_ar.groupby('Due date', as_index=False)['Due'].sum().rename(columns={'Due': 'เงินสดรับ'})
    score_daily = plan_ap.groupby('Due date', as_index=False)['ผลคะแนน'].mean()
    
    # Merge ทั้งหมดในครั้งเดียว
    df_cash = df_dates.merge(ar_daily, on='Due date', how='left') \
                      .merge(ap_daily, on='Due date', how='left') \
                      .merge(score_daily, on='Due date', how='left') \
                      .fillna(0)
    
    # คำนวณ cumsum แบบ vectorized
    df_cash['เงินสดสุทธิ'] = df_cash['เงินสดรับ'] + df_cash['เงินสดจ่าย']
    df_cash['กระแสเงินสดสะสม'] = opening_cash + loan + df_cash['เงินสดสุทธิ'].cumsum()
    
    return df_cash

def find_negative_periods(df_cash, threshold):
    """หาช่วงเวลาที่กระแสเงินสดติดลบ (optimized)"""
    df_neg = df_cash[df_cash['กระแสเงินสดสะสม'] < threshold].copy()
    
    if df_neg.empty:
        return pd.DataFrame(columns=['ช่วงเริ่มต้น', 'ช่วงสิ้นสุด'])
    
    df_neg = df_neg.sort_values('Due date').reset_index(drop=True)
    df_neg['date_diff'] = df_neg['Due date'].diff().dt.days
    df_neg['new_period'] = (df_neg['date_diff'] > 1) | (df_neg['date_diff'].isna())
    df_neg['period_id'] = df_neg['new_period'].cumsum()
    
    periods = df_neg.groupby('period_id')['Due date'].agg(['min', 'max']).reset_index()
    periods.columns = ['period_id', 'ช่วงเริ่มต้น', 'ช่วงสิ้นสุด']
    
    return periods[['ช่วงเริ่มต้น', 'ช่วงสิ้นสุด']]

def extract_and_score_payments(plan_ap, df_periods):
    """แยกและให้คะแนนการจ่ายเงิน (optimized)"""
    if df_periods.empty:
        return pd.DataFrame()
    
    # ใช้ merge แทน loop
    plan_ap = plan_ap.copy()
    
    # สร้าง date range สำหรับแต่ละ period
    period_ranges = []
    for _, period in df_periods.iterrows():
        dates = pd.date_range(period['ช่วงเริ่มต้น'], period['ช่วงสิ้นสุด'], freq='D')
        period_df = pd.DataFrame({
            'Due date': [d.date() for d in dates],
            'ช่วงเริ่มต้น': period['ช่วงเริ่มต้น'],
            'ช่วงสิ้นสุด': period['ช่วงสิ้นสุด']
        })
        period_ranges.append(period_df)
    
    if not period_ranges:
        return pd.DataFrame()
    
    all_periods = pd.concat(period_ranges, ignore_index=True)
    df_payments = plan_ap.merge(all_periods, on='Due date', how='inner')
    
    if df_payments.empty:
        return pd.DataFrame()
    
    # คำนวณ Z-score และ Grade แบบ vectorized
    def calc_zscore_grade(group):
        if len(group) < 2 or group['Due'].std(ddof=0) == 0:
            group['Z_score'] = 0
            group['Grade'] = 2
        else:
            z = (group['Due'] - group['Due'].mean()) / group['Due'].std(ddof=0)
            group['Z_score'] = z
            bins = np.quantile(z, [0, 0.25, 0.5, 0.75, 1])
            group['Grade'] = np.digitize(z + 1, bins, right=True)
        return group
    
    df_scored = df_payments.groupby('ช่วงเริ่มต้น', group_keys=False).apply(calc_zscore_grade)
    return df_scored.reset_index(drop=True)

def adjust_payment_schedule(df_scored, score_range=None, grade_range=None):
    """ปรับตารางการจ่ายเงิน (optimized)"""
    if df_scored is None or df_scored.empty:
        return pd.DataFrame()
    
    df = df_scored.copy()
    mask = pd.Series(True, index=df.index)
    
    if score_range is not None:
        mask &= df['ผลคะแนน'].between(score_range[0], score_range[1], inclusive='both')
    
    if grade_range is not None:
        mask &= df['Grade'].between(grade_range[0], grade_range[1], inclusive='both')
    
    df_adjusted = df[mask].copy()
    if df_adjusted.empty:
        return df_adjusted
    
    df_adjusted['New Due date'] = pd.to_datetime(df_adjusted['ช่วงสิ้นสุด']) + pd.Timedelta(days=1)
    df_adjusted['New Due date'] = df_adjusted['New Due date'].dt.date
    
    return df_adjusted

# ===================== MAIN CONTENT =====================
st.title("💰 CASHFLOW MANAGEMENT SYSTEM")

# สรุปการตั้งค่า
st.markdown("### 📋 สรุปการตั้งค่าปัจจุบัน")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("💵 เงินสดยกมา", f"{เงินสดยกมา:,.0f}")
with col2:
    st.metric("Short Term Loan", f"{Short_term_loan:,.0f}")
with col3:
    st.metric("⚠️ Threshold", f"{threshold:,.0f}")
with col4:
    st.metric("📊 Weighting", scenario_choice)
with col5:
    st.metric("📅 Defer", defer_preset)

st.markdown("---")

# คำนวณกระแสเงินสด
df_cash = calculate_daily_cashflow(df_dates, plan_cashflow_ap, plan_cashflow_ar, เงินสดยกมา, Short_term_loan)

# Tab Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 ภาพรวมกระแสเงินสด",
    "⚠️ ความเสี่ยงด้านลูกหนี้",
    "💡 คำแนะนำการเลื่อนเจ้าหนี้",
    "📈 เปรียบเทียบผลลัพธ์",
    "💰 ภาพรวม AR/AP DAYS"
])

with tab1:
    st.subheader("📊 ภาพรวมกระแสเงินสด")
    
    # สถิติสำคัญ
    col1, col2, col3 = st.columns(3)
    min_cash = df_cash['กระแสเงินสดสะสม'].min()
    max_cash = df_cash['กระแสเงินสดสะสม'].max()
    avg_cash = df_cash['กระแสเงินสดสะสม'].mean()
    
    with col1:
        st.metric("💸 เงินสดต่ำสุด", f"{min_cash:,.0f}", 
                 delta=f"{min_cash - threshold:,.0f}" if threshold > 0 else None,
                 delta_color="inverse")
    with col2:
        st.metric("💰 เงินสดสูงสุด", f"{max_cash:,.0f}")
    with col3:
        st.metric("📊 เงินสดเฉลี่ย", f"{avg_cash:,.0f}")
    
    # กราฟ
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df_cash['Due date']),
        y=df_cash['กระแสเงินสดสะสม'],
        mode='lines',
        name='กระแสเงินสดสะสม',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                     annotation_text="Minimum Cash")
    
    fig.update_layout(
        title="กระแสเงินสดสะสม",
        xaxis_title="วันที่",
        yaxis_title="จำนวนเงิน (บาท)",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📄 ดูข้อมูลรายละเอียด"):
        st.dataframe(df_cash, use_container_width=True)

with tab2:
    st.subheader("⚠️ ความเสี่ยงด้านลูกหนี้")
    
    # เตรียมข้อมูล AR Risk
    plan_ar_filtered = plan_cashflow_ar[plan_cashflow_ar['Due date'] > filtered_start].copy()
    plan_ar_filtered['Risk'] = plan_ar_filtered['From'].map(
        dict(zip(ar_Risk['รายชื่อลูกหนี้'], ar_Risk['Risk']))
    ).fillna(1)
    
    # สรุปภาพรวม
    col1, col2, col3 = st.columns(3)
    high_risk = plan_ar_filtered[plan_ar_filtered['Risk'] >= 4]
    
    with col1:
        st.metric("🔴 ลูกหนี้เสี่ยงสูง", len(high_risk))
    with col2:
        st.metric("💰 มูลค่าเสี่ยงสูง", f"{high_risk['Due'].sum():,.0f}")
    with col3:
        total_ar = plan_ar_filtered['Due'].sum()
        risk_pct = (high_risk['Due'].sum() / total_ar * 100) if total_ar > 0 else 0
        st.metric("📊 % มูลค่าเสี่ยงสูง", f"{risk_pct:.1f}%")
    
    # แสดงรายละเอียดเสี่ยงสูง
    if not high_risk.empty:
        st.warning(f"⚠️ พบลูกหนี้เสี่ยงสูง {len(high_risk)} รายการ มูลค่ารวม {high_risk['Due'].sum():,.0f} บาท")
        
        with st.expander("📋 รายละเอียดทั้งหมด"):
            display_df = high_risk[['From', 'Due date', 'Due', 'Risk']].sort_values('Due date')
            display_df['Due'] = display_df['Due'].apply(lambda x: f"{x:,.0f}")
            st.dataframe(display_df, use_container_width=True)
    else:
        st.success("✅ ไม่พบลูกหนี้ที่มีความเสี่ยงสูง")
    
    # กราฟการกระจายความเสี่ยง
    st.markdown("---")
    st.markdown("#### 📊 การกระจายความเสี่ยง")
    
    col1, col2 = st.columns(2)
    risk_labels = {1: 'ต่ำมาก', 2: 'ต่ำ', 3: 'ปานกลาง', 4: 'สูง', 5: 'สูงมาก'}
    colors = ['#00cc66', '#99cc00', '#ffcc00', '#ff9933', '#ff3333']
    
    with col1:
        risk_count = plan_ar_filtered.groupby('Risk').size().reset_index(name='count')
        risk_count['Risk_Label'] = risk_count['Risk'].map(risk_labels)
        
        fig1 = go.Figure(data=[go.Pie(
            labels=risk_count['Risk_Label'],
            values=risk_count['count'],
            hole=0.3,
            marker=dict(colors=colors)
        )])
        fig1.update_layout(title="จำนวนรายการตามระดับความเสี่ยง")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        risk_amount = plan_ar_filtered.groupby('Risk')['Due'].sum().reset_index()
        risk_amount['Risk_Label'] = risk_amount['Risk'].map(risk_labels)
        
        fig2 = go.Figure(data=[go.Pie(
            labels=risk_amount['Risk_Label'],
            values=risk_amount['Due'],
            hole=0.3,
            marker=dict(colors=colors)
        )])
        fig2.update_layout(title="มูลค่าตามระดับความเสี่ยง")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("💡 คำแนะนำการเลื่อนการจ่ายเงิน")
    
    df_periods = find_negative_periods(df_cash, threshold)
    
    if df_periods.empty:
        st.success("✅ ไม่พบช่วงเวลาที่กระแสเงินสดต่ำกว่า Threshold")
    else:
        st.warning(f"⚠️ พบ {len(df_periods)} ช่วงเวลาที่มีปัญหา")
        st.dataframe(df_periods, use_container_width=True)
        
        df_scored = extract_and_score_payments(plan_cashflow_ap, df_periods)
        
        if not df_scored.empty:
            st.markdown("#### 📋 รายการจ่ายเงินในช่วงที่มีปัญหา")
            summary = df_scored.groupby('ช่วงเริ่มต้น').agg({
                'Due': ['count', 'sum'],
                'ผลคะแนน': 'mean'
            }).round(2)
            summary.columns = ['จำนวนรายการ', 'ยอดรวม', 'คะแนนเฉลี่ย']
            st.dataframe(summary, use_container_width=True)
            
            df_adjusted = adjust_payment_schedule(df_scored, score_range, grade_range)
            
            if not df_adjusted.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📝 จำนวนรายการที่แนะนำให้เลื่อน", len(df_adjusted))
                with col2:
                    st.metric("💵 ยอดเงินรวมที่เลื่อน", f"{df_adjusted['Due'].abs().sum():,.0f}")
                
                selected_cols = ["From", "Category", "Date", "Due date", "New Due date", "Due", "ผลคะแนน", "Grade"]
                df_show = df_adjusted[selected_cols].copy()
                df_show['Due'] = df_show['Due'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
                with st.expander("📄 ดูข้อมูลรายละเอียด"):
                    st.dataframe(df_show, use_container_width=True)
                
                # กราฟแยกตาม Category
                cat_summary = df_adjusted.groupby('Category')['Due'].sum().abs().reset_index()
                fig3 = go.Figure(data=[go.Pie(
                    labels=cat_summary['Category'],
                    values=cat_summary['Due'],
                    hole=0.3
                )])
                fig3.update_layout(title="มูลค่าตาม Category")
                st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("📈 เปรียบเทียบกระแสเงินสดก่อนและหลังปรับปรุง")

    df_periods = find_negative_periods(df_cash, threshold)
    df_scored = extract_and_score_payments(plan_cashflow_ap, df_periods)
    df_adjusted = adjust_payment_schedule(df_scored, score_range, grade_range)

    if not df_adjusted.empty:
        plan_ap_new = plan_cashflow_ap.copy()

        new_dates = df_adjusted.set_index(['From', 'Due date'])['New Due date'].to_dict()

        def update_due_date(row):
            key = (row['From'], row['Due date'])
            return new_dates.get(key, row['Due date'])

        plan_ap_new['Due date'] = plan_ap_new.apply(update_due_date, axis=1)

        df_cash_new = calculate_daily_cashflow(
            df_dates, plan_ap_new, plan_cashflow_ar, เงินสดยกมา, Short_term_loan
        )

        comparison = df_cash[['Due date', 'กระแสเงินสดสะสม']].rename(
            columns={'กระแสเงินสดสะสม': 'แผนเดิม'}
        ).merge(
            df_cash_new[['Due date', 'กระแสเงินสดสะสม']].rename(
                columns={'กระแสเงินสดสะสม': 'แผนปรับปรุง'}
            ),
            on='Due date'
        )

        old_min = comparison['แผนเดิม'].min()
        new_min = comparison['แผนปรับปรุง'].min()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("เงินสดต่ำสุด (แผนเดิม)", f"{old_min:,.0f}")
        with col2:
            st.metric("เงินสดต่ำสุด (แผนปรับปรุง)", f"{new_min:,.0f}",
                    delta=f"{new_min - old_min:,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(comparison['Due date']),
            y=comparison['แผนเดิม'],
            name='แผนเดิม',
            line=dict(color='lightblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(comparison['Due date']),
            y=comparison['แผนปรับปรุง'],
            name='แผนปรับปรุง',
            line=dict(color='green', width=2)
        ))

        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                        annotation_text="Minimum Cash")

        # เพิ่มพื้นที่ AR Risk สูง (ใส่ตรงนี้)
        if not high_risk.empty:
            filtered_dates = pd.to_datetime(high_risk['Due date']).drop_duplicates().sort_values()
            for d in filtered_dates:
                fig.add_vrect(
                    x0=d - pd.Timedelta(days=1), x1=d,
                    fillcolor="red", opacity=0.2, line_width=0,
                    annotation_text="High AR Risk", annotation_position="top left"
                )

        fig.update_layout(
            title="เปรียบเทียบกระแสเงินสดก่อนและหลังปรับปรุง",
            xaxis_title="วันที่",
            yaxis_title="จำนวนเงิน (บาท)",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 ดูตารางเปรียบเทียบ"):
            st.dataframe(comparison, use_container_width=True)

    else:
        old_min = df_cash['กระแสเงินสดสะสม'].min()
        st.metric("เงินสดต่ำสุด (แผนเดิม)", f"{old_min:,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df_cash['Due date']),
            y=df_cash['กระแสเงินสดสะสม'],
            name='แผนเดิม',
            line=dict(color='lightblue', width=2)
        ))

        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                        annotation_text="Minimum Cash")

        # เพิ่มพื้นที่ AR Risk สูง (ใส่ตรงนี้ด้วย)
        if not high_risk.empty:
            filtered_dates = pd.to_datetime(high_risk['Due date']).drop_duplicates().sort_values()
            for d in filtered_dates:
                fig.add_vrect(
                    x0=d - pd.Timedelta(days=1), x1=d,
                    fillcolor="red", opacity=0.2, line_width=0,
                    annotation_text="High AR Risk", annotation_position="top left"
                )

        fig.update_layout(
            title="กระแสเงินสดแผนเดิม (ไม่มีรายการที่ต้องปรับปรุง)",
            xaxis_title="วันที่",
            yaxis_title="จำนวนเงิน (บาท)",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("ไม่พบรายการที่ต้องปรับปรุงตามเกณฑ์ที่กำหนด")

with tab5:
    st.subheader("💰 ภาพรวม AR/AP DAYS")
    
    # =============== AP Analysis ===============
    # แปลงวันที่เป็น datetime และจัดการค่า null
    plan_cashflow_ap['วันที่จ่ายจริง'] = pd.to_datetime(
        plan_cashflow_ap['วันที่จ่ายจริง'], 
        errors='coerce'
    )
    plan_cashflow_ap['Date'] = pd.to_datetime(
        plan_cashflow_ap['Date'], 
        errors='coerce'
    )
    
    # กรองเฉพาะแถวที่มีข้อมูลครบถ้วน
    ap_valid = plan_cashflow_ap.dropna(subset=['วันที่จ่ายจริง', 'Date', 'From'])
    
    # คำนวณจำนวนวันล่าช้า
    ap_valid['delay_days'] = (
        ap_valid['วันที่จ่ายจริง'] - ap_valid['Date']
    ).dt.days
    
    # คำนวณค่าเฉลี่ยความล่าช้าตาม From
    avg_delay_by_from = (
        ap_valid
        .groupby('From', as_index=False)['delay_days']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'avg_delay_days', 'count': 'transactions'})
        .assign(avg_delay_days=lambda x: np.ceil(x['avg_delay_days']).astype(int))
        .sort_values(by='avg_delay_days', ascending=False)
    )
    
    # =============== AR Analysis ===============
    # แปลงวันที่เป็น datetime และจัดการค่า null
    plan_cashflow_ar['วันที่รับเงินจริง'] = pd.to_datetime(
    plan_cashflow_ar['วันที่รับเงินจริง'],
    dayfirst=True,
    errors='coerce'
    )
    
    plan_cashflow_ar['Date'] = pd.to_datetime(
        plan_cashflow_ar['Date'],
        format='%d/%m/%Y',
        errors='coerce'
    )
    # กรองเฉพาะแถวที่มีข้อมูลครบถ้วน
    ar_valid = plan_cashflow_ar.dropna(subset=['วันที่รับเงินจริง', 'Date', 'From'])
    
    # คำนวณจำนวนวันล่าช้า
    ar_valid['delay_days'] = (
        ar_valid['วันที่รับเงินจริง'] - ar_valid['Date']
    ).dt.days
    # คำนวณค่าเฉลี่ยความล่าช้าตาม From
    avg_delay_ar_by_from = (
        ar_valid
        .groupby('From', as_index=False)['delay_days']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'avg_delay_days', 'count': 'transactions'})
        .assign(avg_delay_days=lambda x: np.ceil(x['avg_delay_days']).astype(int))
        .sort_values(by='avg_delay_days', ascending=False)
    )
    
    # =============== Display in 2 Columns ===============
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💸 เจ้าหนี้การค้า (AP)")
        st.markdown(f"**จำนวนรายการทั้งหมด:** {len(ap_valid):,}")
        st.markdown(f"**ค่าเฉลี่ยความล่าช้าโดยรวม:** {ap_valid['delay_days'].mean():.1f} วัน")
        st.dataframe(
            avg_delay_by_from,
            use_container_width=True,
            hide_index=True,
            column_config={
                "From": "ผู้ขาย/เจ้าหนี้",
                "avg_delay_days": st.column_config.NumberColumn(
                    "ค่าเฉลี่ยวันล่าช้า",
                    format="%d วัน"
                ),
                "transactions": st.column_config.NumberColumn(
                    "จำนวนรายการ",
                    format="%d"
                )
            }
        )
    
    with col2:
        st.markdown("### 💰 ลูกหนี้การค้า (AR)")
        st.markdown(f"**จำนวนรายการทั้งหมด:** {len(ar_valid):,}")
        st.markdown(f"**ค่าเฉลี่ยความล่าช้าโดยรวม:** {ar_valid['delay_days'].mean():.1f} วัน")
        st.dataframe(
            avg_delay_ar_by_from,
            use_container_width=True,
            hide_index=True,
            column_config={
                "From": "ลูกค้า/ลูกหนี้",
                "avg_delay_days": st.column_config.NumberColumn(
                    "ค่าเฉลี่ยวันล่าช้า",
                    format="%d วัน"
                ),
                "transactions": st.column_config.NumberColumn(
                    "จำนวนรายการ",
                    format="%d"
                )
            }
        )
