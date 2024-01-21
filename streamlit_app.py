import streamlit as st
import numpy as np
import pandas as pd
import torch

st.title('🎈 Streamlit Forum Explorer')

@st.cache_data
def load_df():
  return pd.read_csv('streamlit_forum_16Jan2024.csv')

@st.cache_data
def load_embeddings():
  return torch.load(f'corpus_embeddings_16Jan2024.pt')

@st.cache_data
def load_cluster_topics():
  return pd.read_csv('cluster_topics.csv', header=False)

@st.cache_data
def load_tsne_2d_vectors():
  return np.load('tsne_2d_vectors.npy')

df = load_df()
corpus_embeddings = load_embeddings()
cluster_topics = load_cluster_topics()
tsne_2d_vectors = load_tsne_2d_vectors()
