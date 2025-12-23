"""
President Dallin H. Oaks Talk Analysis - Streamlit App (Simplified Starter Version)
This is a simpler version to get you started quickly
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import string
from io import BytesIO
import time
import os
import json

# Page config
st.set_page_config(
    page_title="Oaks Talk Analysis",
    page_icon="📖",
    layout="wide"
)

# Download NLTK data if needed
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    return True

download_nltk_data()

# Cache talk data in session state
if 'talks_data' not in st.session_state:
    st.session_state.talks_data = []
    st.session_state.all_text = ""

# Stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
    'should', 'now', 'elder', 'president', 'brother', 'sister', 'saint', 'saints',
    'conference', 'general', 'talk', 'spoke', 'speaking', 'said', 'says', 'today',
    'time', 'year', 'years', 'day', 'days', 'may', 'might', 'must', 'shall',
    'would', 'could', 'also', 'even', 'well', 'much', 'many', 'every', 'first',
    'second', 'third', 'last', 'next', 'one', 'two', 'three', 'four', 'five',
    'lord', 'god', 'jesus', 'christ', 'church', 'lds', 'latter', 'brethren',
    'sisters', 'oaks', 'dallin', 'thank', 'name', 'amen', 'things', 'thing',
    'make', 'made', 'way', 'us', 'let', 'like', 'come', 'came', 'go', 'went'
}

# Helper functions
def clean_text(text):
    """Clean text for processing"""
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text):
    """Tokenize text and remove stopwords"""
    text = clean_text(text)
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation and any(c.isalpha() for c in t)]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

def generate_wordcloud_simple(text, title="Word Cloud"):
    """Generate a simple word cloud"""
    if not text:
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=STOPWORDS,
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def fetch_sample_talks():
    """Fetch a sample of talks from bencrowder.net"""
    url = "https://bencrowder.net/collected-talks/dallin-h-oaks/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        talks = []
        content_div = soup.find('div', class_='entry-content') or soup.find('article')
        
        if content_div:
            links = content_div.find_all('a')
            for link in links[:20]:  # Limit to 20 for demo
                href = link.get('href', '')
                if href and ('churchofjesuschrist.org' in href or 'speeches.byu.edu' in href):
                    talks.append({
                        'title': link.get_text(strip=True),
                        'url': href
                    })
        return talks
    except:
        return []

def download_talk_content(url):
    """Download content from a talk URL"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find content
        article = soup.find('article') or soup.find('div', class_='body-block') or soup.find('main')
        
        if article:
            for element in article(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            text = article.get_text(separator=' ', strip=True)
            return re.sub(r'\s+', ' ', text)
    except:
        pass
    
    return None

# Sidebar
with st.sidebar:
    st.title("📖 Talk Analysis")
    
    st.markdown("""
    ### Quick Start:
    1. Load sample talks
    2. Generate word clouds
    3. Analyze custom themes
    """)
    
    st.divider()
    
    # Display stats
    if st.session_state.talks_data:
        st.metric("Talks Loaded", len(st.session_state.talks_data))
        total_words = len(st.session_state.all_text.split())
        st.metric("Total Words", f"{total_words:,}")
    else:
        st.info("No talks loaded yet")
    
    st.divider()
    
    # Load sample data button
    if st.button("🔄 Load Sample Talks", type="primary"):
        with st.spinner("Fetching talks..."):
            talks = fetch_sample_talks()
            
            if talks:
                progress_bar = st.progress(0)
                successful = 0
                
                for i, talk in enumerate(talks[:10]):  # Limit to 10 for speed
                    content = download_talk_content(talk['url'])
                    if content:
                        st.session_state.talks_data.append({
                            'title': talk['title'],
                            'content': content
                        })
                        successful += 1
                    
                    progress_bar.progress((i + 1) / min(len(talks), 10))
                    time.sleep(1)  # Rate limiting
                
                # Combine all text
                st.session_state.all_text = ' '.join([t['content'] for t in st.session_state.talks_data])
                
                st.success(f"Loaded {successful} talks!")
                st.rerun()
    
    # Clear data button
    if st.button("🗑️ Clear Data"):
        st.session_state.talks_data = []
        st.session_state.all_text = ""
        st.rerun()

# Main content
st.title("President Dallin H. Oaks Talk Analysis")
st.markdown("### Simple Text Analysis Dashboard")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "☁️ Word Clouds", "🔍 Custom Analysis", "📈 Frequency"])

with tab1:
    st.header("Overview")
    
    if st.session_state.talks_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Talks", len(st.session_state.talks_data))
        
        with col2:
            total_words = len(st.session_state.all_text.split())
            st.metric("Total Words", f"{total_words:,}")
        
        with col3:
            avg_words = total_words // len(st.session_state.talks_data) if st.session_state.talks_data else 0
            st.metric("Avg Words/Talk", f"{avg_words:,}")
        
        st.subheader("Loaded Talks")
        
        talks_df = pd.DataFrame([
            {'Title': t['title'], 'Word Count': len(t['content'].split())}
            for t in st.session_state.talks_data
        ])
        
        st.dataframe(talks_df, use_container_width=True)
        
    else:
        st.info("👈 Click 'Load Sample Talks' in the sidebar to get started!")

with tab2:
    st.header("Word Cloud Generator")
    
    if st.session_state.all_text:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Options")
            
            cloud_type = st.radio("Select Type", ["All Talks", "Single Talk", "Custom Theme"])
            
            if cloud_type == "Single Talk":
                selected_talk = st.selectbox(
                    "Select Talk",
                    [t['title'] for t in st.session_state.talks_data]
                )
            
            elif cloud_type == "Custom Theme":
                theme_name = st.text_input("Theme Name", placeholder="e.g., Faith")
                keywords = st.text_area(
                    "Keywords (comma-separated)",
                    placeholder="faith, believe, testimony, witness"
                )
        
        with col2:
            if st.button("Generate Word Cloud"):
                with st.spinner("Generating..."):
                    if cloud_type == "All Talks":
                        buf = generate_wordcloud_simple(st.session_state.all_text, "All Talks")
                        if buf:
                            st.image(buf, use_column_width=True)
                    
                    elif cloud_type == "Single Talk":
                        talk = next(t for t in st.session_state.talks_data if t['title'] == selected_talk)
                        buf = generate_wordcloud_simple(talk['content'], selected_talk[:50])
                        if buf:
                            st.image(buf, use_column_width=True)
                    
                    elif cloud_type == "Custom Theme" and theme_name and keywords:
                        # Extract sentences with keywords
                        keyword_list = [k.strip() for k in keywords.split(',')]
                        theme_sentences = []
                        
                        for talk in st.session_state.talks_data:
                            sentences = sent_tokenize(talk['content'])
                            for sentence in sentences:
                                if any(kw.lower() in sentence.lower() for kw in keyword_list):
                                    theme_sentences.append(sentence)
                        
                        if theme_sentences:
                            theme_text = ' '.join(theme_sentences)
                            buf = generate_wordcloud_simple(theme_text, f"Theme: {theme_name}")
                            if buf:
                                st.image(buf, use_column_width=True)
                                st.success(f"Found {len(theme_sentences)} sentences about {theme_name}")
                        else:
                            st.warning("No sentences found with those keywords")
    else:
        st.info("Load talks first to generate word clouds")

with tab3:
    st.header("Custom Theme Analysis")
    
    if st.session_state.all_text:
        st.markdown("Search for specific themes or concepts across all talks")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            search_term = st.text_input("Search Term", placeholder="Enter a word or phrase")
            
            if search_term and st.button("Search"):
                search_lower = search_term.lower()
                results = []
                
                for talk in st.session_state.talks_data:
                    sentences = sent_tokenize(talk['content'])
                    matching_sentences = [s for s in sentences if search_lower in s.lower()]
                    
                    if matching_sentences:
                        results.append({
                            'Talk': talk['title'],
                            'Mentions': len(matching_sentences),
                            'Sample': matching_sentences[0][:200] + '...'
                        })
                
                if results:
                    st.success(f"Found '{search_term}' in {len(results)} talks")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.warning(f"No mentions of '{search_term}' found")
        
        with col2:
            st.subheader("Quick Theme Analysis")
            
            themes = {
                'Faith': ['faith', 'believe', 'testimony', 'trust'],
                'Family': ['family', 'marriage', 'children', 'parent'],
                'Service': ['service', 'serve', 'help', 'minister'],
                'Prayer': ['prayer', 'pray', 'ask', 'answers'],
                'Scriptures': ['scripture', 'bible', 'book', 'mormon']
            }
            
            selected_theme = st.selectbox("Select Theme", list(themes.keys()))
            
            if st.button("Analyze Theme"):
                keywords = themes[selected_theme]
                count = 0
                
                for kw in keywords:
                    count += st.session_state.all_text.lower().count(kw.lower())
                
                st.metric(f"{selected_theme} Mentions", count)
                
                # Show word distribution
                word_counts = {kw: st.session_state.all_text.lower().count(kw.lower()) 
                              for kw in keywords}
                
                df_theme = pd.DataFrame(list(word_counts.items()), 
                                       columns=['Word', 'Count'])
                st.bar_chart(df_theme.set_index('Word'))
    else:
        st.info("Load talks first to analyze themes")

with tab4:
    st.header("Word Frequency Analysis")
    
    if st.session_state.all_text:
        st.subheader("Most Frequent Words")
        
        # Get word frequencies
        tokens = tokenize(st.session_state.all_text)
        word_freq = Counter(tokens)
        
        # Number of words to display
        n_words = st.slider("Number of words", 10, 50, 20)
        
        top_words = word_freq.most_common(n_words)
        
        # Create dataframe
        df_freq = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
        total = sum(word_freq.values())
        df_freq['Percentage'] = (df_freq['Frequency'] / total * 100).round(2)
        
        # Display table
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df_freq, use_container_width=True)
        
        with col2:
            # Bar chart of top 10
            st.bar_chart(df_freq.head(10).set_index('Word')['Frequency'])
        
        # Download button
        csv = df_freq.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="word_frequencies.csv",
            mime="text/csv"
        )
    else:
        st.info("Load talks first to analyze word frequencies")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>President Dallin H. Oaks Talk Analysis | Built with Streamlit</p>
    <p>Data source: <a href='https://bencrowder.net/collected-talks/dallin-h-oaks/' target='_blank'>bencrowder.net</a></p>
</div>
""", unsafe_allow_html=True)
