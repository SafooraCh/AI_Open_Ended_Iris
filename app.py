"""
Streamlit App for Random Forest Iris Classifier
This version trains the model on-the-fly if it doesn't exist
Deploy this app using: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide"
)

# Load or train the model
@st.cache_resource
def load_model():
    model_path = 'random_forest_model.pkl'
    
    # Check if model exists
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except:
            pass
    
    # Train new model if file doesn't exist or can't be loaded
    st.info("Training model... This will only happen once!")
    
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Save model for future use
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    except:
        pass  # If we can't save, that's okay
    
    return model

# Load model
model = load_model()

# Load iris data for reference
iris = load_iris()

# Title and description
st.title("🌸 Iris Flower Classification")
st.markdown("""
This app uses a **Random Forest Classifier** to predict the species of Iris flowers.
Enter the flower measurements below to get a prediction!
""")

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📊 Input Features")
    st.markdown("Adjust the sliders to set flower measurements:")
    
    # Input sliders
    sepal_length = st.slider(
        "Sepal Length (cm)", 
        float(iris.data[:, 0].min()), 
        float(iris.data[:, 0].max()), 
        float(iris.data[:, 0].mean()),
        help="Length of the sepal in centimeters"
    )
    
    sepal_width = st.slider(
        "Sepal Width (cm)", 
        float(iris.data[:, 1].min()), 
        float(iris.data[:, 1].max()), 
        float(iris.data[:, 1].mean()),
        help="Width of the sepal in centimeters"
    )
    
    petal_length = st.slider(
        "Petal Length (cm)", 
        float(iris.data[:, 2].min()), 
        float(iris.data[:, 2].max()), 
        float(iris.data[:, 2].mean()),
        help="Length of the petal in centimeters"
    )
    
    petal_width = st.slider(
        "Petal Width (cm)", 
        float(iris.data[:, 3].min()), 
        float(iris.data[:, 3].max()), 
        float(iris.data[:, 3].mean()),
        help="Width of the petal in centimeters"
    )
    
    # Predict button
    predict_btn = st.button("🔮 Predict Species", type="primary", use_container_width=True)

with col2:
    # Create input dataframe
    input_data = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })
    
    # Display input
    st.header("📋 Current Measurements")
    st.dataframe(input_data, use_container_width=True, hide_index=True)
    
    # Make prediction automatically or on button click
    if predict_btn or True:  # Auto-predict on slider change
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Display prediction
        st.header("🎯 Prediction Result")
        
        # Create metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Predicted Species", iris.target_names[prediction[0]], 
                     help="The predicted iris species")
        
        with metric_col2:
            st.metric("Confidence", f"{max(prediction_proba[0])*100:.2f}%",
                     help="Model's confidence in the prediction")
        
        with metric_col3:
            st.metric("Model", "Random Forest",
                     help="100 decision trees")
        
        # Display probabilities
        st.subheader("📈 Class Probabilities")
        
        proba_df = pd.DataFrame({
            'Species': iris.target_names,
            'Probability': prediction_proba[0] * 100
        }).sort_values('Probability', ascending=False)
        
        # Create a nice bar chart
        import plotly.express as px
        
        fig = px.bar(
            proba_df, 
            x='Probability', 
            y='Species',
            orientation='h',
            text='Probability',
            color='Probability',
            color_continuous_scale='Viridis',
            labels={'Probability': 'Probability (%)'}
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# Expandable sections
with st.expander("ℹ️ About This App"):
    st.markdown("""
    ### How It Works
    
    This application uses a **Random Forest Classifier** trained on the famous Iris dataset.
    
    **Random Forest** is an ensemble learning method that:
    - Creates multiple decision trees
    - Each tree votes on the classification
    - The final prediction is the majority vote
    
    ### Model Details
    - **Algorithm**: Random Forest Classifier
    - **Number of Trees**: 100
    - **Training Data**: Iris Dataset (150 samples)
    - **Features**: 4 (sepal & petal measurements)
    - **Classes**: 3 (Setosa, Versicolor, Virginica)
    
    ### Performance Metrics
    - **Accuracy**: ~97%
    - **Precision**: ~97%
    - **Recall**: ~97%
    - **F1-Score**: ~97%
    """)

with st.expander("📚 About the Iris Dataset"):
    st.markdown("""
    ### The Iris Dataset
    
    The Iris flower dataset is one of the most famous datasets in machine learning.
    
    **Features:**
    - **Sepal Length**: Length of the sepal (outer part of the flower)
    - **Sepal Width**: Width of the sepal
    - **Petal Length**: Length of the petal (inner part of the flower)
    - **Petal Width**: Width of the petal
    
    **Species:**
    - **Setosa**: Usually has smaller petals
    - **Versicolor**: Medium-sized petals
    - **Virginica**: Larger petals
    
    **Dataset Statistics:**
    - Total samples: 150
    - Samples per class: 50
    - Created by: Ronald Fisher (1936)
    """)

with st.expander("🔍 Sample Data"):
    # Show sample data
    sample_df = pd.DataFrame(
        iris.data[:10], 
        columns=iris.feature_names
    )
    sample_df['species'] = [iris.target_names[i] for i in iris.target[:10]]
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", 
             caption="Iris Versicolor", use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🎯 Quick Test Examples")
    
    if st.button("🌺 Setosa Example", use_container_width=True):
        st.session_state.sepal_length = 5.1
        st.session_state.sepal_width = 3.5
        st.session_state.petal_length = 1.4
        st.session_state.petal_width = 0.2
    
    if st.button("🌸 Versicolor Example", use_container_width=True):
        st.session_state.sepal_length = 6.4
        st.session_state.sepal_width = 3.2
        st.session_state.petal_length = 4.5
        st.session_state.petal_width = 1.5
    
    if st.button("🌹 Virginica Example", use_container_width=True):
        st.session_state.sepal_length = 6.3
        st.session_state.sepal_width = 3.3
        st.session_state.petal_length = 6.0
        st.session_state.petal_width = 2.5
    
    st.markdown("---")
    
    st.subheader("📊 Model Info")
    st.info(f"""
    **Trees**: {model.n_estimators}  
    **Max Depth**: {model.max_depth}  
    **Features**: {len(iris.feature_names)}  
    **Classes**: {len(iris.target_names)}
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit 🎈 | Model: Random Forest Classifier</p>
    <p>Lab 12 - Task 2: Machine Learning Classification</p>
</div>
""", unsafe_allow_html=True)
