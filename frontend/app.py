import streamlit as st
import requests

st.title('Semantic Match Calculator')

# Two big text boxes
text_a = st.text_area('Text A', height=200, placeholder='Paste or type text here...')
text_b = st.text_area('Text B', height=200, placeholder='Paste or type text here...')

# Dropdowns
search_mode = st.selectbox('Choose Semantic Mode', ['Keyword Search', 'Semantic Search', 'Hybrid Search'])
vector_db = st.selectbox('Choose Vector DB', ['FAISS', 'Qdrant'])

# Embedding model selection (for semantic/hybrid)
embedding_models = [
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2',
    'paraphrase-MiniLM-L6-v2',
    # Add more as needed
]
embedding_model = st.selectbox('Choose Embedding Model', embedding_models, index=0)

# Map UI values to backend values
search_mode_map = {
    'Keyword Search': 'keyword',
    'Semantic Search': 'semantic',
    'Hybrid Search': 'hybrid',
}

# Score box (placeholder)
st.markdown('---')
st.subheader('Semantic Match Score')
score = st.empty()

if st.button('Calculate Similarity'):
    if not text_a.strip() or not text_b.strip():
        st.warning('Please enter text in both boxes.')
    else:
        try:
            payload = {
                'text_a': text_a,
                'text_b': text_b,
                'search_mode': search_mode_map[search_mode],
                'embedding_model': embedding_model,
                'vector_db': vector_db,
            }
            response = requests.post('http://localhost:8000/similarity', json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if search_mode == 'Hybrid Search' and 'debug' in data:
                    debug = data['debug']
                    st.metric(label='Hybrid Score', value=f"{data['score']:.2f}")
                    st.metric(label='Keyword Score', value=f"{debug.get('score_keyword', '--'):.2f}")
                    st.metric(label='Semantic Score', value=f"{debug.get('score_semantic', '--'):.2f}")
                else:
                    score.metric(label='Score', value=f"{data['score']:.2f}")
                if 'debug' in data:
                    with st.expander('Debug Info'):
                        st.json(data['debug'])
            else:
                score.metric(label='Score', value='--')
                st.error(f"Backend error: {response.status_code}")
        except Exception as e:
            score.metric(label='Score', value='--')
            st.error(f"Could not connect to backend: {e}")
else:
    score.metric(label='Score', value='--') 