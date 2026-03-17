import streamlit as st
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.title("Memory Page Compressibility Predictor")

df = pd.read_csv("phase2_stage1_v2_results.csv")

features = [
"feat_distinct",
"feat_max_freq_ratio",
"feat_zero_ratio",
"feat_run_count"
]

X = df[features]
y = df["label"].map({"compressible":1,"incompressible":0})

model = RandomForestClassifier()
model.fit(X,y)

uploaded_file = st.file_uploader("Upload memory page (.bin)", type=["bin"])

def extract_features(sample):
    counts = Counter(sample)
    distinct = len(counts)
    max_freq_ratio = max(counts.values())/len(sample)
    zero_ratio = counts.get(0,0)/len(sample)
    run_count = sum(1 for i in range(1,len(sample)) if sample[i]==sample[i-1])
    
    return [distinct,max_freq_ratio,zero_ratio,run_count]

if uploaded_file is not None:
    
    page_bytes = uploaded_file.read()
    sample = page_bytes[:128]
    
    feats = extract_features(sample)
    
    prediction = model.predict([feats])[0]
    
    st.write("### Extracted Features")
    
    st.write("Distinct Bytes:",feats[0])
    st.write("Max Frequency Ratio:",feats[1])
    st.write("Zero Ratio:",feats[2])
    st.write("Run Count:",feats[3])
    
    if prediction==1:
        st.success("Prediction: COMPRESSIBLE")
    else:
        st.error("Prediction: INCOMPRESSIBLE")