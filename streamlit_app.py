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


# Chart
alt.data_transformers.disable_max_rows()

colors = [
  '#fff100', # yellow
  '#ff8c00', # orange
  '#e81123', # red
  '#ec008c', # magenta
  '#68217a', # purple
  '#00188f', # blue
  '#00bcf2', # cyan
  '#00b294', # teal
  '#009e49', # green
  '#bad80a' # lime
]

def my_theme():
  return {
    'config': {
      'view': {'continuousHeight': 400, 'continuousWidth': 400},  # from the default theme
      'range': {'category': colors}
    }
  }
alt.themes.register('my_theme', my_theme)
alt.themes.enable('my_theme')

tsne_plot = alt.Chart(df_cluster).mark_circle(size=60).encode(
                x=alt.X('x:Q', axis=alt.Axis(title='Dimension 1', titlePadding=12, titleFontSize=16, titleFontWeight=900)),
                y=alt.Y('y:Q', axis=alt.Axis(title='Dimension 2', titlePadding=12, titleFontSize=16, titleFontWeight=900)),
                # x='x:Q',
                # y='y:Q',
                color='cluster:N',
                opacity=alt.value(0.3),
                tooltip=['title', 'cluster']
            )

# tsne_plot


