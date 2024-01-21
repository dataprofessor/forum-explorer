import streamlit as st
import torch

st.title('🎈 Streamlit Forum Explorer')

@st.cache_data
def load_df():
  return pd.read_csv('streamlit_forum_16Jan2024.csv')

@st.cache_data
def load_embeddings():
  return torch.load(f'corpus_embeddings_16Jan2024.pt')

df = load_df()
corpus_embeddings = load_embeddings()
