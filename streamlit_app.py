import streamlit as st
import altair as alt
import joblib
import numpy as np
from openTSNE import TSNE
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

st.title('🎈 Streamlit Forum Explorer')

# Load data
@st.cache_data
def load_df():
  return pd.read_csv('data/streamlit_forum_16Jan2024.csv')

@st.cache_data
def load_embeddings():
  return torch.load('data/corpus_embeddings_16Jan2024.pt')

@st.cache_data
def load_cluster_topics():
  return pd.read_csv('data/cluster_topics.csv', header=None)

@st.cache_data
def load_tsne_2d_vectors():
  return np.load('data/tsne_corpus_embeddings.npy')

@st.cache_data
def load_tsne_posts_vectors_clusters():
  return pd.read_csv('data/tsne_posts_vectors_clusters.csv')

# pre-trained tSNE model
def load_corpus_embeddings():
  return joblib.load('data/tsne_corpus_embeddings.sav')

df = load_df()
corpus_embeddings = load_embeddings()
cluster_topics = load_cluster_topics()
tsne_2d_vectors = load_tsne_2d_vectors()
df_cluster = load_tsne_posts_vectors_clusters()
loaded_tsne = load_corpus_embeddings()

# Parameters
with st.sidebar:
  k_neighbors = st.slider('How many nearest neighbors?', 1, 100, 5)

# Query
st.markdown('#### Query')
input_query = st.text_input('Ask a question about Streamlit', placeholder='Enter your question here ...')

# Generate embeddings for query
embedder = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = embedder.encode(input_query, convert_to_tensor=True)
corpus = list(df.title)

# Find top scoring nearest neighbors
top_k = min(k_neighbors, len(corpus_embeddings))
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
top_results = torch.topk(cos_scores, k=top_k)

if input_query != '':
  st.markdown('#### Results')
  st.warning(f'**{k_neighbors} Nearest neighbors for:** {input_query}', icon='📍')

  for score, idx in zip(top_results[0], top_results[1]):
    post_link = f"https://discuss.streamlit.io/t/{df.slug[idx.item()]}/{df.id[idx.item()]}"
    st.write(f"- [{corpus[idx]}]({post_link})", "`(Score: {:.3f})`".format(score))
    



##########
# Chart rendering
st.markdown('#### Topic clusters')
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
      'view': {'continuousHeight': 400},
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

st.altair_chart(tsne_plot, use_container_width=True)
