import streamlit as st
import numpy as np
import pandas as pd
import torch

st.title('🎈 Streamlit Forum Explorer')

# Load data

#@st.cache_data
def load_df():
  return pd.read_csv('data/streamlit_forum_16Jan2024.csv')

#@st.cache_data
def load_embeddings():
  return torch.load('data/corpus_embeddings_16Jan2024.pt')

#@st.cache_data
def load_cluster_topics():
  return pd.read_csv('data/cluster_topics.csv', header=None)

#@st.cache_data
def load_tsne_2d_vectors():
  return np.load('data/tsne_2d_vectors.npy')

#@st.cache_data
def load_tsne_posts_vectors_clusters():
  return pd.read_csv('tsne_posts_vectors_clusters.csv')

df = load_df()
corpus_embeddings = load_embeddings()
cluster_topics = load_cluster_topics()
tsne_2d_vectors = load_tsne_2d_vectors()
df_cluster = load_tsne_posts_vectors_clusters()

