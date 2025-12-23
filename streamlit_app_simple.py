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

# Try to import demo data
try:
    from demo_data import DEMO_TALKS, get_demo_talks, get_combined_demo_text
    demo_data_available = True
except ImportError:
    demo_data_available = False
    # Fallback demo data if import fails
    DEMO_TALKS = [
        {
            'title': 'Demo Talk: Faith and Trust',
            'content': 'Faith is the foundation of our spiritual lives. We must trust in the Lord. ' * 100
        },
        {
            'title': 'Demo Talk: Family and Love', 
            'content': 'Families are central to God plan. Love brings happiness. ' * 100
        }
    ]

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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        talks = []
        
        # Try multiple possible content containers
        content_div = soup.find('div', class_='entry-content')
        if not content_div:
            content_div = soup.find('article')
        if not content_div:
            content_div = soup.find('main')
        if not content_div:
            content_div = soup.find('div', id='content')
        
        if content_div:
            # Find all links
            all_links = content_div.find_all('a', href=True)
            
            for link in all_links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Check for talk URLs (church sites and BYU)
                if any(domain in href for domain in ['churchofjesuschrist.org', 'speeches.byu.edu', 'lds.org']):
                    # Skip if it's just a domain link without specific content
                    if not href.endswith('/') and len(text) > 10:
                        talks.append({
                            'title': text,
                            'url': href
                        })
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_talks = []
        for talk in talks:
            if talk['url'] not in seen_urls:
                seen_urls.add(talk['url'])
                unique_talks.append(talk)
        
        return unique_talks[:30]  # Return up to 30 talks
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching talk list: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return []

def download_talk_content(url):
    """Download content from a talk URL"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        # Add timeout and better error handling
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Different selectors for different sites
        article = None
        
        # Church website selectors
        if 'churchofjesuschrist.org' in url:
            article = soup.find('article')
            if not article:
                article = soup.find('div', class_='body-block')
            if not article:
                article = soup.find('div', id='content')
            if not article:
                article = soup.find('section', class_='article-content')
        
        # BYU speeches selectors
        elif 'speeches.byu.edu' in url:
            article = soup.find('div', class_='transcript')
            if not article:
                article = soup.find('div', class_='speech-text')
            if not article:
                article = soup.find('article')
            if not article:
                article = soup.find('div', class_='content')
        
        # Generic fallback
        if not article:
            article = soup.find('main')
            if not article:
                article = soup.find('article')
                
        if article:
            # Remove unwanted elements
            for element in article(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Get text
            text = article.get_text(separator=' ', strip=True)
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n+', ' ', text)
            
            # Only return if we have substantial content
            if len(text) > 500:
                return text
                
    except requests.exceptions.Timeout:
        st.warning(f"Timeout loading: {url[:50]}...")
    except requests.exceptions.RequestException as e:
        st.warning(f"Error loading talk: {str(e)[:50]}")
    except Exception as e:
        st.warning(f"Unexpected error: {str(e)[:50]}")
    
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
        with st.spinner("Fetching talk list from bencrowder.net..."):
            talks = fetch_sample_talks()
            
            if talks:
                st.success(f"Found {len(talks)} talks!")
                
                # Let user choose how many to load
                num_to_load = st.slider("How many talks to load?", 1, min(len(talks), 20), 5)
                
                if st.button(f"Download {num_to_load} Talks"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    successful = 0
                    failed = 0
                    
                    for i, talk in enumerate(talks[:num_to_load]):
                        status_text.text(f"Loading: {talk['title'][:40]}...")
                        content = download_talk_content(talk['url'])
                        
                        if content:
                            st.session_state.talks_data.append({
                                'title': talk['title'],
                                'content': content
                            })
                            successful += 1
                        else:
                            failed += 1
                        
                        progress_bar.progress((i + 1) / num_to_load)
                        time.sleep(0.5)  # Rate limiting
                    
                    # Combine all text
                    if st.session_state.talks_data:
                        st.session_state.all_text = ' '.join([t['content'] for t in st.session_state.talks_data])
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    if successful > 0:
                        st.success(f"✅ Loaded {successful} talks successfully!")
                        if failed > 0:
                            st.warning(f"⚠️ {failed} talks failed to load")
                        st.rerun()
                    else:
                        st.error("Failed to load any talks. Try again or use demo data.")
            else:
                st.error("Could not fetch talk list. Check your internet connection.")
                
                # Offer demo data as fallback
                st.info("💡 Try loading demo data instead!")
                
    # Demo data button (always visible)
    if st.button("📚 Load Demo Data", type="secondary"):
        # Load comprehensive demo talks
        st.session_state.talks_data = DEMO_TALKS
        st.session_state.all_text = ' '.join([t['content'] for t in DEMO_TALKS])
        st.success(f"Loaded {len(DEMO_TALKS)} demo talks for testing!")
        st.info("These are excerpts from real talks for demonstration purposes.")
        st.rerun()
    
    # Manual text input option
    with st.expander("➕ Add Talk Manually"):
        manual_title = st.text_input("Talk Title")
        manual_content = st.text_area("Talk Content", height=200)
        
        if st.button("Add Manual Talk"):
            if manual_title and manual_content:
                st.session_state.talks_data.append({
                    'title': manual_title,
                    'content': manual_content
                })
                st.session_state.all_text = ' '.join([t['content'] for t in st.session_state.talks_data])
                st.success(f"Added '{manual_title}'!")
                st.rerun()
            else:
                st.error("Please provide both title and content")
    
    # Clear data button
    if st.button("🗑️ Clear All Data"):
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
            {'Title': t['title'][:60], 'Word Count': len(t['content'].split())}
            for t in st.session_state.talks_data
        ])
        
        st.dataframe(talks_df, use_container_width=True, hide_index=True)
        
        # Show sample content
        with st.expander("View Sample Content"):
            if st.session_state.talks_data:
                sample_talk = st.session_state.talks_data[0]
                st.write(f"**{sample_talk['title']}**")
                st.write(sample_talk['content'][:500] + "...")
        
    else:
        st.info("👈 Click 'Load Sample Talks' in the sidebar to get started!")
        
        # Quick instructions
        st.markdown("""
        ### Getting Started:
        
        1. **Load Talks** - Click the button in the sidebar to fetch talks
        2. **Generate Word Clouds** - Visualize the most common words
        3. **Search Themes** - Find specific concepts across all talks
        4. **Analyze Frequencies** - See statistical word analysis
        
        ### Alternative Options:
        - **Demo Data** - If loading fails, use the demo data option
        - **Manual Entry** - Add talks manually using the sidebar expander
        - **Custom Themes** - Search for any word or phrase once talks are loaded
        """)

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
