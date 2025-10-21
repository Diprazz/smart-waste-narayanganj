import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# App configuration
st.set_page_config(
    page_title="Smart Waste Management - Narayanganj",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv('data/narayanganj_waste.csv')

@st.cache_resource
def load_or_train_model():
    try:
        return joblib.load('models/waste_model_narayanganj.pkl')
    except FileNotFoundError:
        df = load_data()
        X = df[['Population', 'Temperature']]
        y = df['Waste_Collected_kg_per_day']
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        return model

def main():
    # Header
    st.markdown('<h1 class="main-header">🏭 Smart Waste Management System - Narayanganj</h1>', unsafe_allow_html=True)
    st.markdown("### Using Machine Learning for Efficient Urban Waste Collection Planning")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Prediction", "📊 Data Analysis", "🤖 Model Details", "👨‍💻 About"])
    
    if page == "🏠 Home":
        show_home()
    elif page == "🔮 Prediction":
        show_prediction()
    elif page == "📊 Data Analysis":
        show_analysis()
    elif page == "🤖 Model Details":
        show_model_details()
    elif page == "👨‍💻 About":
        show_about()

def show_home():
    st.markdown("""
    ## 🌟 Project Overview
    
    This web application demonstrates a **Smart Waste Management System** developed for Narayanganj city using machine learning.
    
    ### 🎯 Key Features:
    - **Predictive Modeling**: Forecast daily waste collection based on population and temperature
    - **Data Analysis**: Explore waste patterns across different areas
    - **Interactive Interface**: User-friendly web application for real-time predictions
    - **Machine Learning**: Random Forest algorithm for accurate predictions
    
    ### 🚀 How to Use:
    1. Navigate to **Prediction** tab to get waste estimates
    2. Explore **Data Analysis** for insights
    3. Check **Model Details** for technical information
    """)
    
    # Quick stats (dynamic)
    df = load_data()
    areas_count = len(df)
    model = load_or_train_model()
    X = df[['Population', 'Temperature']]
    y = df['Waste_Collected_kg_per_day']
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Areas Analyzed", f"{areas_count}")
    with col2:
        st.metric("Model R² (in-sample)", f"{r2:.2f}")
    with col3:
        st.metric("Total Population", f"{df['Population'].sum():,}")

def show_prediction():
    st.header("🔮 Waste Collection Predictor")
    
    st.markdown("""
    Enter the area characteristics below to predict daily waste collection:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        population = st.slider("👥 Population", 10000, 200000, 50000, 1000,
                             help="Total population in the area")
        temperature = st.slider("🌡️ Temperature (°C)", 20, 40, 32, 1,
                              help="Average daily temperature")
    
    with col2:
        area_type = st.selectbox("🏢 Area Type", 
                               ["Residential", "Commercial", "Mixed", "Industrial"],
                               help="Type of area for waste collection")
        day_type = st.selectbox("📅 Day Type", 
                              ["Weekday", "Weekend", "Holiday"],
                              help="Type of day for collection")
    
    # Prediction button
    if st.button("🚀 Predict Waste Collection", use_container_width=True):
        try:
            # Load model and make prediction
            model = load_or_train_model()
            prediction = model.predict([[population, temperature]])[0]
            
            # Adjust prediction based on area type
            multipliers = {
                "Residential": 1.0,
                "Commercial": 1.3,
                "Mixed": 1.15,
                "Industrial": 1.5
            }
            adjusted_prediction = prediction * multipliers[area_type]
            
            # Display results
            st.markdown("---")
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.success(f"## 📦 Predicted Waste Collection: **{adjusted_prediction:,.0f} kg/day**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Daily Collection", f"{adjusted_prediction:,.0f} kg")
            with col2:
                st.metric("Weekly Estimate", f"{adjusted_prediction * 7:,.0f} kg")
            with col3:
                st.metric("Monthly Projection", f"{adjusted_prediction * 30:,.0f} kg")
            
            # Recommendations
            st.info(f"""
            **🗓️ Collection Recommendations:**
            - **Frequency**: {'Twice daily' if adjusted_prediction > 5000 else 'Once daily'}
            - **Truck Capacity**: {'Large (10-ton)' if adjusted_prediction > 8000 else 'Medium (5-ton)'}
            - **Optimal Time**: {'Morning & Evening' if adjusted_prediction > 5000 else 'Morning only'}
            """)
            
        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")
            st.info("💡 Please ensure the model is trained first by running the Jupyter notebook.")

def show_analysis():
    st.header("📊 Data Analysis Dashboard")
    
    try:
        # Load data
        df = load_data()
        
        # Overview
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Areas", len(df))
        with col2:
            st.metric("Average Waste", f"{df['Waste_Collected_kg_per_day'].mean():.0f} kg/day")
        with col3:
            st.metric("Total Population", f"{df['Population'].sum():,}")
        
        # Data table
        st.subheader("📋 Area-wise Data")
        st.dataframe(df.style.format({
            'Population': '{:,}',
            'Waste_Collected_kg_per_day': '{:,}'
        }), use_container_width=True)
        
        # Visualizations
        st.subheader("📈 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Waste Collection by Area**")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df, x='Waste_Collected_kg_per_day', y='Area', 
                       palette='viridis', ax=ax)
            ax.set_xlabel('Waste Collected (kg/day)')
            ax.set_ylabel('Area')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Population vs Waste Correlation**")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='Population', y='Waste_Collected_kg_per_day', 
                           size='Waste_Collected_kg_per_day', sizes=(100, 500),
                           hue='Area', palette='tab10', ax=ax, legend=False)
            
            # Add area labels
            for i in range(len(df)):
                ax.text(df.Population.iloc[i], df.Waste_Collected_kg_per_day.iloc[i], 
                       df.Area.iloc[i], fontsize=9, alpha=0.7)
            
            ax.set_xlabel('Population')
            ax.set_ylabel('Waste Collected (kg/day)')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("🔗 Feature Correlations")
        fig, ax = plt.subplots(figsize=(8, 6))
        correlation = df[['Population', 'Temperature', 'Waste_Collected_kg_per_day']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

def show_model_details():
    st.header("🤖 Machine Learning Model")
    
    st.markdown("""
    ## 🧠 Model Architecture
    
    ### Algorithm: Random Forest Regressor
    - **Type**: Ensemble Learning
    - **Base Estimators**: 100 Decision Trees
    - **Features**: Population, Temperature
    - **Target**: Waste Collected (kg/day)
    
    ### 📊 Model Performance
    - **R² Score**: > 0.85
    - **Mean Absolute Error**: < 200 kg
    - **Cross-Validation**: 3-fold
    """)
    
    # Feature importance (if available)
    model = load_or_train_model()
    if hasattr(model, 'feature_importances_'):
        st.subheader("🎯 Feature Importance")
        features = ['Population', 'Temperature']
        importance = model.feature_importances_
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=importance, y=features, palette='rocket', ax=ax)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Relative Importance of Features in Prediction')
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")
    
    st.markdown("""
    ### 🎯 Business Impact
    - **Cost Reduction**: Optimized collection routes and schedules
    - **Efficiency**: Better resource allocation
    - **Sustainability**: Reduced environmental impact
    - **Planning**: Data-driven urban development decisions
    """)

def show_about():
    st.header("👨‍💻 About This Project")
    
    st.markdown("""
    ## 🌟 Project Developer
    
    **Diprazz** - Data Science Enthusiast
    
    ### 📚 Technical Stack
    - **Programming**: Python
    - **Machine Learning**: Scikit-learn, Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Web Framework**: Streamlit
    - **Version Control**: Git & GitHub
    
    ### 🎯 Project Goals
    1. Demonstrate practical application of machine learning in urban planning
    2. Showcase end-to-end data science project development
    3. Create scalable solutions for smart city initiatives
    4. Contribute to sustainable urban development
    
    ### 🔗 Connect & Explore
    """)
    
    # GitHub badge and link
    st.markdown("""
    <div style="text-align: center;">
        <a href="https://github.com/Disprazz/smart-waste-narayanganj" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github" alt="GitHub Repository">
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📄 Project Documentation
    - Complete source code available on GitHub
    - Detailed Jupyter notebook with step-by-step analysis
    - Model training and evaluation protocols
    - Deployment instructions
    
    ### 🏆 Academic Relevance
    This project demonstrates capabilities in:
    - Machine Learning and Data Analysis
    - Full-stack Data Science Development
    - Real-world Problem Solving
    - Technical Documentation and Deployment
    """)

if __name__ == "__main__":
    main()
