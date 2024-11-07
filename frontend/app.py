#frontend/app.py (Streamlit)
import streamlit as st
import requests
import folium
from streamlit_folium import folium_static
import json
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="LULC Classification", layout="wide")

def create_map(data):
    """Create a folium map with the classification results"""
    # Create base map centered on ROI
    m = folium.Map(location=[data['center_lat'], data['center_lon']], zoom_start=12)
    
    # Add classification overlay
    for tile, pred in zip(data['tiles'], data['predictions']):
        color = data['colors'][pred]
        folium.GeoJson(
            tile,
            style_function=lambda x, color=color: {
                'fillColor': color,
                'fillOpacity': 0.7,
                'color': 'none'
            }
        ).add_to(m)
    
    return m

def create_chart(class_percentages):
    """Create a bar chart of class percentages"""
    df = pd.DataFrame({
        'Class': list(class_percentages.keys()),
        'Percentage': list(class_percentages.values())
    })
    
    fig = px.bar(df, x='Class', y='Percentage',
                 title='Land Use / Land Cover Distribution')
    return fig

st.title("🌍 LULC Classification Assistant")

with st.sidebar:
    st.header("Analysis Parameters")
    
    # Input fields for analysis
    iso_code = st.selectbox(
        "Country Code",
        ["IND", "NPL", "BGD", "BTN", "LKA"],  # South Asian countries
        index=0,
        help="ISO 3166-1 alpha-3 country code"
    )
    
    adm_level = st.selectbox(
        "Administrative Level",
        ["ADM0", "ADM1", "ADM2"],
        index=1,
        help="ADM0: Country, ADM1: State/Province, ADM2: District"
    )
    
    # Only show ROI input if not at country level
    if adm_level != "ADM0":
        roi = st.text_input(
            "Region of Interest",
            placeholder="e.g., Tamil Nadu for ADM1, Vellore for ADM2",
            help="Enter the name of the region you want to analyze"
        )
    else:
        roi = iso_code  # For country-level analysis
    
    analyze_button = st.button("Analyze Region", type="primary")

# Initialize chat history for follow-up discussion
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.analysis_complete = False

# Handle analysis
if analyze_button:
    with st.spinner("Analyzing region..."):
        try:
            response = requests.post(
                "http://localhost:8000/classify/",
                json={
                    "iso_code": iso_code,
                    "adm_level": adm_level,
                    "roi": roi
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Store analysis results in session state
            st.session_state.analysis_data = data
            st.session_state.analysis_complete = True
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Summary", "Visualization", "Detailed Analysis"])
            
            with tab1:
                st.markdown(f"### Analysis for {data['roi']}")
                st.markdown(data['summary'])
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Classification Map")
                    m = create_map(data)
                    folium_static(m)
                
                with col2:
                    st.subheader("Distribution")
                    fig = create_chart(data['class_percentages'])
                    st.plotly_chart(fig)
            
            with tab3:
                st.markdown("### Detailed Class Distribution")
                for cls, percentage in data['class_percentages'].items():
                    st.metric(cls, f"{percentage:.1f}%")
            
            # Add initial analysis to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I've completed the analysis for {data['roi']}. " + data['summary']
            })
            
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'json'):
                error_detail = e.response.json().get('detail', str(e))
                st.error(f"Analysis Error: {error_detail}")
            else:
                st.error(f"Error connecting to server: {str(e)}")

# Chat interface for follow-up questions (only shown after analysis)
if st.session_state.analysis_complete:
    st.divider()
    st.subheader("Ask Follow-up Questions")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the analysis results..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                response = requests.post(
                    "http://localhost:8000/chat/",
                    json={
                        "prompt": prompt,
                        "analysis_data": st.session_state.analysis_data
                    }
                )
                response.raise_for_status()
                reply = response.json()["response"]
                st.markdown(reply)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply
                })
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")


