import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Cleaning & Reporting Automation",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #e91e63;
        font-weight: 700;
    }
    .cleaning-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .issue-found {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 10px;
        margin: 5px 0;
    }
    .issue-fixed {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🧹 Data Cleaning & Reporting Automation")
st.markdown("**Automated data preprocessing, cleaning, and report generation**")
st.markdown("---")

# Function to load data
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format!")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Data quality checker
def analyze_data_quality(df):
    """Analyze data quality issues"""
    issues = {
        'missing_values': {},
        'duplicates': 0,
        'data_types': {},
        'outliers': {},
        'inconsistencies': {},
        'total_rows': len(df),
        'total_columns': len(df.columns)
    }
    
    # Missing values
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            issues['missing_values'][col] = {
                'count': missing,
                'percentage': (missing / len(df)) * 100
            }
    
    # Duplicates
    issues['duplicates'] = df.duplicated().sum()
    
    # Data types
    for col in df.columns:
        issues['data_types'][col] = str(df[col].dtype)
    
    # Outliers (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            issues['outliers'][col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100
            }
    
    return issues

# Automated cleaning functions
def clean_missing_values(df, method='drop', fill_value=None):
    """Handle missing values"""
    df_cleaned = df.copy()
    
    if method == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif method == 'fill_mean':
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    elif method == 'fill_median':
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
    elif method == 'fill_mode':
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown', inplace=True)
    elif method == 'fill_custom' and fill_value is not None:
        df_cleaned = df_cleaned.fillna(fill_value)
    elif method == 'forward_fill':
        df_cleaned = df_cleaned.fillna(method='ffill')
    elif method == 'backward_fill':
        df_cleaned = df_cleaned.fillna(method='bfill')
    
    return df_cleaned

def remove_duplicates(df, subset=None):
    """Remove duplicate rows"""
    return df.drop_duplicates(subset=subset, keep='first')

def handle_outliers(df, method='remove', columns=None):
    """Handle outliers"""
    df_cleaned = df.copy()
    
    if columns is None:
        columns = df_cleaned.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'remove':
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        elif method == 'cap':
            df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)
    
    return df_cleaned

def standardize_text(df, columns=None):
    """Standardize text data"""
    df_cleaned = df.copy()
    
    if columns is None:
        columns = df_cleaned.select_dtypes(include=['object']).columns
    
    for col in columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.title()
    
    return df_cleaned

def convert_data_types(df, conversions):
    """Convert data types"""
    df_cleaned = df.copy()
    
    for col, dtype in conversions.items():
        try:
            if dtype == 'datetime':
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
            elif dtype == 'numeric':
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            elif dtype == 'category':
                df_cleaned[col] = df_cleaned[col].astype('category')
            elif dtype == 'string':
                df_cleaned[col] = df_cleaned[col].astype(str)
        except:
            pass
    
    return df_cleaned

# Generate automated report
def generate_report(df_original, df_cleaned, cleaning_steps):
    """Generate comprehensive cleaning report"""
    report = {
        'original_shape': df_original.shape,
        'cleaned_shape': df_cleaned.shape,
        'rows_removed': df_original.shape[0] - df_cleaned.shape[0],
        'columns_removed': df_original.shape[1] - df_cleaned.shape[1],
        'cleaning_steps': cleaning_steps,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return report

# Sidebar
st.sidebar.header("📁 Data Import")
uploaded_file = st.sidebar.file_uploader(
    "Upload data file (CSV/Excel)",
    type=['csv', 'xlsx', 'xls']
)

df_original = None
if uploaded_file is not None:
    df_original = load_data(uploaded_file)
    if df_original is not None:
        st.sidebar.success(f"✅ Loaded {len(df_original)} rows, {len(df_original.columns)} columns")
else:
    st.sidebar.warning("⚠️ Please upload a file to get started")

# Main analysis
if df_original is not None:
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔍 Quality Analysis", 
        "🧹 Data Cleaning", 
        "📈 Cleaned Data", 
        "📋 Automated Report"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("📊 Original Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(df_original):,}")
        
        with col2:
            st.metric("Total Columns", f"{len(df_original.columns):,}")
        
        with col3:
            memory_usage = df_original.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
        
        with col4:
            numeric_cols = len(df_original.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", f"{numeric_cols}")
        
        st.markdown("---")
        
        # Display sample data
        st.subheader("📋 Data Preview (First 10 rows)")
        st.dataframe(df_original.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Column information
        st.subheader("📑 Column Information")
        
        col_info = pd.DataFrame({
            'Column': df_original.columns,
            'Data Type': df_original.dtypes.values,
            'Non-Null Count': df_original.count().values,
            'Null Count': df_original.isnull().sum().values,
            'Null %': (df_original.isnull().sum() / len(df_original) * 100).round(2).values,
            'Unique Values': [df_original[col].nunique() for col in df_original.columns]
        })
        
        st.dataframe(col_info, use_container_width=True, height=400)
    
    # Tab 2: Quality Analysis
    with tab2:
        st.header("🔍 Data Quality Analysis")
        
        # Analyze data quality
        issues = analyze_data_quality(df_original)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_missing = sum([v['count'] for v in issues['missing_values'].values()])
            st.metric("Missing Values", f"{total_missing:,}", 
                     delta=f"{len(issues['missing_values'])} columns affected" if total_missing > 0 else "None")
        
        with col2:
            st.metric("Duplicate Rows", f"{issues['duplicates']:,}",
                     delta=f"{(issues['duplicates']/issues['total_rows']*100):.1f}%" if issues['duplicates'] > 0 else "None")
        
        with col3:
            total_outliers = sum([v['count'] for v in issues['outliers'].values()])
            st.metric("Outliers Detected", f"{total_outliers:,}",
                     delta=f"{len(issues['outliers'])} columns" if total_outliers > 0 else "None")
        
        with col4:
            data_quality = 100 - ((total_missing + issues['duplicates']) / (issues['total_rows'] * issues['total_columns']) * 100)
            st.metric("Data Quality Score", f"{data_quality:.1f}%")
        
        st.markdown("---")
        
        # Missing values details
        if issues['missing_values']:
            st.subheader("❌ Missing Values Details")
            
            missing_df = pd.DataFrame([
                {
                    'Column': col,
                    'Missing Count': info['count'],
                    'Missing %': f"{info['percentage']:.2f}%"
                }
                for col, info in issues['missing_values'].items()
            ]).sort_values('Missing Count', ascending=False)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(missing_df, use_container_width=True, height=300)
            
            with col2:
                fig_missing = px.bar(
                    missing_df,
                    x='Column',
                    y='Missing Count',
                    title='Missing Values by Column',
                    color='Missing Count',
                    color_continuous_scale='Reds'
                )
                fig_missing.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
        
        st.markdown("---")
        
        # Duplicates
        if issues['duplicates'] > 0:
            st.subheader("🔄 Duplicate Rows")
            st.markdown(f"""
            <div class='issue-found'>
                <strong>⚠️ Found {issues['duplicates']} duplicate rows ({(issues['duplicates']/issues['total_rows']*100):.2f}%)</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample duplicates
            duplicates = df_original[df_original.duplicated(keep=False)].head(10)
            st.dataframe(duplicates, use_container_width=True)
        else:
            st.success("✅ No duplicate rows found!")
        
        st.markdown("---")
        
        # Outliers
        if issues['outliers']:
            st.subheader("📊 Outliers Detected")
            
            outlier_df = pd.DataFrame([
                {
                    'Column': col,
                    'Outlier Count': info['count'],
                    'Outlier %': f"{info['percentage']:.2f}%"
                }
                for col, info in issues['outliers'].items()
            ]).sort_values('Outlier Count', ascending=False)
            
            st.dataframe(outlier_df, use_container_width=True)
            
            # Box plots for outliers
            st.subheader("📈 Outlier Visualization")
            selected_col = st.selectbox("Select column to visualize:", list(issues['outliers'].keys()))
            
            fig_box = px.box(
                df_original,
                y=selected_col,
                title=f'Box Plot - {selected_col}',
                color_discrete_sequence=['#e91e63']
            )
            fig_box.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.success("✅ No significant outliers detected!")
    
    # Tab 3: Data Cleaning
    with tab3:
        st.header("🧹 Automated Data Cleaning")
        
        # Initialize session state for cleaned data
        if 'df_cleaned' not in st.session_state:
            st.session_state.df_cleaned = df_original.copy()
            st.session_state.cleaning_steps = []
        
        # Cleaning options
        st.subheader("🔧 Cleaning Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 1️⃣ Handle Missing Values")
            missing_method = st.selectbox(
                "Select method:",
                ["Select method", "Drop rows with missing values", "Fill with mean", 
                 "Fill with median", "Fill with mode", "Forward fill", "Backward fill"]
            )
            
            if st.button("Apply Missing Values Cleaning", key="missing"):
                if missing_method != "Select method":
                    method_map = {
                        "Drop rows with missing values": "drop",
                        "Fill with mean": "fill_mean",
                        "Fill with median": "fill_median",
                        "Fill with mode": "fill_mode",
                        "Forward fill": "forward_fill",
                        "Backward fill": "backward_fill"
                    }
                    
                    before_rows = len(st.session_state.df_cleaned)
                    st.session_state.df_cleaned = clean_missing_values(
                        st.session_state.df_cleaned, 
                        method=method_map[missing_method]
                    )
                    after_rows = len(st.session_state.df_cleaned)
                    
                    step = f"Handled missing values using {missing_method}"
                    st.session_state.cleaning_steps.append({
                        'step': step,
                        'rows_before': before_rows,
                        'rows_after': after_rows,
                        'rows_affected': before_rows - after_rows
                    })
                    
                    st.success(f"✅ {step}")
                    st.rerun()
        
        with col2:
            st.markdown("### 2️⃣ Remove Duplicates")
            if st.button("Remove Duplicate Rows", key="duplicates"):
                before_rows = len(st.session_state.df_cleaned)
                duplicates_found = st.session_state.df_cleaned.duplicated().sum()
                
                st.session_state.df_cleaned = remove_duplicates(st.session_state.df_cleaned)
                after_rows = len(st.session_state.df_cleaned)
                
                step = f"Removed {duplicates_found} duplicate rows"
                st.session_state.cleaning_steps.append({
                    'step': step,
                    'rows_before': before_rows,
                    'rows_after': after_rows,
                    'rows_affected': before_rows - after_rows
                })
                
                st.success(f"✅ {step}")
                st.rerun()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 3️⃣ Handle Outliers")
            outlier_method = st.selectbox(
                "Select method:",
                ["Select method", "Remove outliers", "Cap outliers"]
            )
            
            if st.button("Apply Outlier Handling", key="outliers"):
                if outlier_method != "Select method":
                    before_rows = len(st.session_state.df_cleaned)
                    method = 'remove' if outlier_method == "Remove outliers" else 'cap'
                    
                    st.session_state.df_cleaned = handle_outliers(
                        st.session_state.df_cleaned,
                        method=method
                    )
                    after_rows = len(st.session_state.df_cleaned)
                    
                    step = f"Handled outliers using {outlier_method}"
                    st.session_state.cleaning_steps.append({
                        'step': step,
                        'rows_before': before_rows,
                        'rows_after': after_rows,
                        'rows_affected': before_rows - after_rows
                    })
                    
                    st.success(f"✅ {step}")
                    st.rerun()
        
        with col4:
            st.markdown("### 4️⃣ Standardize Text")
            if st.button("Standardize Text Columns", key="text"):
                before_rows = len(st.session_state.df_cleaned)
                
                st.session_state.df_cleaned = standardize_text(st.session_state.df_cleaned)
                
                step = "Standardized text columns (trim, title case)"
                st.session_state.cleaning_steps.append({
                    'step': step,
                    'rows_before': before_rows,
                    'rows_after': before_rows,
                    'rows_affected': 0
                })
                
                st.success(f"✅ {step}")
                st.rerun()
        
        st.markdown("---")
        
        # Reset and complete
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset All Cleaning", type="secondary"):
                st.session_state.df_cleaned = df_original.copy()
                st.session_state.cleaning_steps = []
                st.success("✅ Reset to original data")
                st.rerun()
        
        with col2:
            if st.button("✅ Complete Cleaning", type="primary"):
                st.session_state.cleaning_complete = True
                st.success("✅ Data cleaning completed!")
                st.balloons()
        
        st.markdown("---")
        
        # Cleaning history
        if st.session_state.cleaning_steps:
            st.subheader("📜 Cleaning History")
            
            for i, step in enumerate(st.session_state.cleaning_steps, 1):
                st.markdown(f"""
                <div class='issue-fixed'>
                    <strong>{i}. {step['step']}</strong><br>
                    Rows: {step['rows_before']:,} → {step['rows_after']:,} 
                    ({step['rows_affected']:,} affected)
                </div>
                """, unsafe_allow_html=True)
    
    # Tab 4: Cleaned Data
    with tab4:
        st.header("📈 Cleaned Data Overview")
        
        df_display = st.session_state.df_cleaned if 'df_cleaned' in st.session_state else df_original
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(df_display):,}",
                     delta=f"{len(df_display) - len(df_original):,} rows")
        
        with col2:
            st.metric("Total Columns", f"{len(df_display.columns):,}")
        
        with col3:
            missing_after = df_display.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_after:,}",
                     delta=f"-{df_original.isnull().sum().sum() - missing_after:,}")
        
        with col4:
            duplicates_after = df_display.duplicated().sum()
            st.metric("Duplicates", f"{duplicates_after:,}",
                     delta=f"-{df_original.duplicated().sum() - duplicates_after:,}")
        
        st.markdown("---")
        
        # Display cleaned data
        st.subheader("📋 Cleaned Data Preview")
        st.dataframe(df_display, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numeric Columns**")
            numeric_summary = df_display.describe()
            st.dataframe(numeric_summary, use_container_width=True)
        
        with col2:
            st.markdown("**Categorical Columns**")
            cat_cols = df_display.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                cat_summary = df_display[cat_cols].describe()
                st.dataframe(cat_summary, use_container_width=True)
            else:
                st.info("No categorical columns found")
        
        st.markdown("---")
        
        # Export cleaned data
        st.subheader("💾 Export Cleaned Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f'cleaned_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
            )
        
        with col2:
            # Excel export
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_display.to_excel(writer, sheet_name='Cleaned Data', index=False)
            
            st.download_button(
                label="📥 Download as Excel",
                data=buffer.getvalue(),
                file_name=f'cleaned_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
        
        with col3:
            # JSON export
            json_data = df_display.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f'cleaned_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json',
            )
    
    # Tab 5: Automated Report
    with tab5:
        st.header("📋 Automated Data Cleaning Report")
        
        df_cleaned = st.session_state.df_cleaned if 'df_cleaned' in st.session_state else df_original
        cleaning_steps = st.session_state.cleaning_steps if 'cleaning_steps' in st.session_state else []
        
        # Generate report
        report = generate_report(df_original, df_cleaned, cleaning_steps)
        
        # Report header
        st.markdown(f"""
        ### Data Cleaning Report
        **Generated:** {report['timestamp']}
        
        ---
        """)
        
        # Summary section
        st.subheader("📊 Executive Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Original Data:**
            - Rows: {report['original_shape'][0]:,}
            - Columns: {report['original_shape'][1]:,}
            - Missing Values: {df_original.isnull().sum().sum():,}
            - Duplicates: {df_original.duplicated().sum():,}
            """)
        
        with col2:
            st.markdown(f"""
            **Cleaned Data:**
            - Rows: {report['cleaned_shape'][0]:,}
            - Columns: {report['cleaned_shape'][1]:,}
            - Missing Values: {df_cleaned.isnull().sum().sum():,}
            - Duplicates: {df_cleaned.duplicated().sum():,}
            """)
        
        st.markdown("---")
        
        # Cleaning operations
        st.subheader("🔧 Cleaning Operations Performed")
        
        if cleaning_steps:
            for i, step in enumerate(cleaning_steps, 1):
                st.markdown(f"""
                **{i}. {step['step']}**
                - Rows before: {step['rows_before']:,}
                - Rows after: {step['rows_after']:,}
                - Rows affected: {step['rows_affected']:,}
                """)
        else:
            st.info("No cleaning operations performed yet")
        
        st.markdown("---")
        
        # Quality improvements
        st.subheader("📈 Data Quality Improvements")
        
        # Calculate improvements
        missing_before = df_original.isnull().sum().sum()
        missing_after = df_cleaned.isnull().sum().sum()
        duplicates_before = df_original.duplicated().sum()
        duplicates_after = df_cleaned.duplicated().sum()
        
        improvements = pd.DataFrame({
            'Metric': ['Missing Values', 'Duplicate Rows', 'Total Rows', 'Data Completeness'],
            'Before': [
                missing_before,
                duplicates_before,
                len(df_original),
                f"{((len(df_original) * len(df_original.columns) - missing_before) / (len(df_original) * len(df_original.columns)) * 100):.2f}%"
            ],
            'After': [
                missing_after,
                duplicates_after,
                len(df_cleaned),
                f"{((len(df_cleaned) * len(df_cleaned.columns) - missing_after) / (len(df_cleaned) * len(df_cleaned.columns)) * 100):.2f}%"
            ],
            'Improvement': [
                f"-{missing_before - missing_after:,}",
                f"-{duplicates_before - duplicates_after:,}",
                f"{len(df_cleaned) - len(df_original):,}",
                "↑"
            ]
        })
        
        st.dataframe(improvements, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📊 Quality Comparison Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing values comparison
            missing_comparison = pd.DataFrame({
                'Stage': ['Before Cleaning', 'After Cleaning'],
                'Missing Values': [missing_before, missing_after]
            })
            
            fig_missing = px.bar(
                missing_comparison,
                x='Stage',
                y='Missing Values',
                title='Missing Values Comparison',
                color='Missing Values',
                color_continuous_scale='Reds'
            )
            fig_missing.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig_missing, use_container_width=True)
        
        with col2:
            # Duplicates comparison
            duplicates_comparison = pd.DataFrame({
                'Stage': ['Before Cleaning', 'After Cleaning'],
                'Duplicates': [duplicates_before, duplicates_after]
            })
            
            fig_dup = px.bar(
                duplicates_comparison,
                x='Stage',
                y='Duplicates',
                title='Duplicates Comparison',
                color='Duplicates',
                color_continuous_scale='Oranges'
            )
            fig_dup.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig_dup, use_container_width=True)
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        remaining_issues = []
        
        if df_cleaned.isnull().sum().sum() > 0:
            remaining_issues.append("- Still has missing values - consider additional cleaning")
        
        if df_cleaned.duplicated().sum() > 0:
            remaining_issues.append("- Still has duplicate rows - review duplicate removal criteria")
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df_cleaned[(df_cleaned[col] < Q1 - 1.5*IQR) | (df_cleaned[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                remaining_issues.append(f"- Column '{col}' still has {len(outliers)} outliers")
        
        if remaining_issues:
            st.warning("**Remaining Issues:**")
            for issue in remaining_issues:
                st.markdown(issue)
        else:
            st.success("✅ **All major data quality issues have been resolved!**")
        
        st.markdown("---")
        
        # Export report
        st.subheader("💾 Export Report")
        
        # Generate text report
        report_text = f"""
DATA CLEANING REPORT
Generated: {report['timestamp']}

==============================================
EXECUTIVE SUMMARY
==============================================

Original Data:
- Rows: {report['original_shape'][0]:,}
- Columns: {report['original_shape'][1]:,}
- Missing Values: {missing_before:,}
- Duplicates: {duplicates_before:,}

Cleaned Data:
- Rows: {report['cleaned_shape'][0]:,}
- Columns: {report['cleaned_shape'][1]:,}
- Missing Values: {missing_after:,}
- Duplicates: {duplicates_after:,}

Data Quality Score: {((len(df_cleaned) * len(df_cleaned.columns) - missing_after) / (len(df_cleaned) * len(df_cleaned.columns)) * 100):.2f}%

==============================================
CLEANING OPERATIONS
==============================================

"""
        
        for i, step in enumerate(cleaning_steps, 1):
            report_text += f"""
{i}. {step['step']}
   - Rows before: {step['rows_before']:,}
   - Rows after: {step['rows_after']:,}
   - Rows affected: {step['rows_affected']:,}
"""
        
        report_text += f"""

==============================================
IMPROVEMENTS
==============================================

- Missing values reduced: {missing_before - missing_after:,} ({((missing_before - missing_after)/missing_before*100 if missing_before > 0 else 0):.1f}%)
- Duplicates removed: {duplicates_before - duplicates_after:,}
- Rows cleaned: {report['rows_removed']:,}

==============================================
END OF REPORT
==============================================
"""
        
        st.download_button(
            label="📥 Download Report (TXT)",
            data=report_text,
            file_name=f'cleaning_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            mime='text/plain',
        )

else:
    # Welcome screen
    st.info("👋 **Welcome to Data Cleaning & Reporting Automation!**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What This Tool Does
        
        Automated data cleaning and quality improvement:
        - **Detect** data quality issues
        - **Clean** missing values, duplicates, outliers
        - **Standardize** text and formats
        - **Generate** automated reports
        
        ### ✨ Key Features
        
        - 🔍 Automated quality analysis
        - 🧹 One-click data cleaning
        - 📊 Visual quality reports
        - 💾 Export cleaned data
        - 📋 Comprehensive reporting
        - 🤖 Workflow automation
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Common Data Issues Handled
        
        | Issue | Solution |
        |-------|----------|
        | Missing Values | Multiple fill strategies |
        | Duplicates | Automatic removal |
        | Outliers | Remove or cap |
        | Text Inconsistency | Standardization |
        | Wrong Data Types | Auto-conversion |
        | Whitespace | Trimming |
        
        ### 🔧 Cleaning Methods
        
        - Drop missing rows
        - Fill with mean/median/mode
        - Forward/backward fill
        - Remove duplicates
        - Handle outliers
        - Standardize text
        """)
    
    st.markdown("---")
    
    st.subheader("📝 Sample Data Format")
    sample_data = pd.DataFrame({
        'ID': [1, 2, 3, 2, 5],
        'Name': ['John ', 'JANE', 'bob', 'JANE', '  Alice'],
        'Age': [25, None, 150, 30, 28],
        'Email': ['john@email.com', 'jane@email.com', None, 'jane@email.com', 'alice@email.com'],
        'Salary': [50000, 60000, 55000, 60000, None]
    })
    st.dataframe(sample_data, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Common issues in this sample:**
    - Row 2 is duplicated (row 4)
    - Missing values in Age and Email
    - Outlier in Age (150)
    - Inconsistent text formatting in Name
    - Whitespace in Name column
    """)
    
    st.info("💡 **Tip:** Upload your messy data file to start automated cleaning!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🧹 Data Cleaning & Reporting Automation | Powered by Python & Pandas</p>
        <p>Data Analyst Internship Project - Task 4</p>
    </div>
    """,
    unsafe_allow_html=True
)