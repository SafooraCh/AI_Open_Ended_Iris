print("\n" + "="*60)
print("CREATING STREAMLIT APP")
print("="*60)

streamlit_app_code = '''"""
Streamlit App for Random Forest Iris Classifier
Deploy this app using: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris

# Page configuration
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Load iris data for reference
iris = load_iris()

# Title and description
st.title("🌸 Iris Flower Classification")
st.markdown("""
This app uses a **Random Forest Classifier** to predict the species of Iris flowers.
Enter the flower measurements below to get a prediction!
""")

# Sidebar for input
st.sidebar.header("Input Features")
st.sidebar.markdown("Adjust the sliders to set flower measurements:")

# Input sliders
sepal_length = st.sidebar.slider(
    "Sepal Length (cm)", 
    float(iris.data[:, 0].min()), 
    float(iris.data[:, 0].max()), 
    float(iris.data[:, 0].mean())
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)", 
    float(iris.data[:, 1].min()), 
    float(iris.data[:, 1].max()), 
    float(iris.data[:, 1].mean())
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)", 
    float(iris.data[:, 2].min()), 
    float(iris.data[:, 2].max()), 
    float(iris.data[:, 2].mean())
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)", 
    float(iris.data[:, 3].min()), 
    float(iris.data[:, 3].max()), 
    float(iris.data[:, 3].mean())
)

# Create input dataframe
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

# Display input
st.subheader("📊 Input Features")
st.dataframe(input_data, use_container_width=True)

# Predict button
if st.button("🔮 Predict Species", type="primary"):
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display prediction
    st.subheader("🎯 Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Species", iris.target_names[prediction[0]])
    
    with col2:
        st.metric("Confidence", f"{max(prediction_proba[0])*100:.2f}%")
    
    with col3:
        st.metric("Model Used", "Random Forest")
    
    # Display probabilities
    st.subheader("📈 Class Probabilities")
    proba_df = pd.DataFrame({
        'Species': iris.target_names,
        'Probability': prediction_proba[0]
    }).sort_values('Probability', ascending=False)
    
    st.bar_chart(proba_df.set_index('Species'))
    
    # Display detailed probabilities
    st.dataframe(proba_df, use_container_width=True)

# Information section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.info("""
This model was trained on the famous Iris dataset using a Random Forest Classifier with 100 trees.

**Model Performance:**
- Accuracy: 97%+
- Precision: 97%+
- Recall: 97%+
- F1-Score: 97%+
""")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Model: Random Forest Classifier")
'''

# Save Streamlit app
with open('app.py', 'w') as f:
    f.write(streamlit_app_code)
print("✓ Streamlit app saved as 'app.py'")