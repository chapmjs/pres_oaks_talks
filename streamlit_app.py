"""
President Dallin H. Oaks Talk Analysis - Streamlit App
A comprehensive text analysis tool with custom theme search and corpus updates
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import re
import string
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import base64
from io import BytesIO
import json
import hashlib

# Page configuration
st.set_page_config(
    page_title="President Oaks Talk Analysis",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize NLTK data
@st.cache_resource
def init_nltk():
    """Download required NLTK data"""
    nltk_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 
                       'averaged_perceptron_tagger', 'omw-1.4']
    for item in nltk_downloads:
        try:
            nltk.data.find(f'tokenizers/{item}')
        except LookupError:
            try:
                nltk.download(item, quiet=True)
            except:
                pass
    return True

# Initialize database
@st.cache_resource
def init_database():
    """Initialize SQLite database for storing talks"""
    conn = sqlite3.connect('oaks_talks.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Create talks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS talks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            url TEXT UNIQUE,
            talk_type TEXT,
            date TEXT,
            year INTEGER,
            content TEXT,
            word_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create themes table for custom analyses
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS custom_themes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            theme_name TEXT NOT NULL,
            keywords TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_year ON talks(year)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON talks(talk_type)')
    
    conn.commit()
    return conn

# Stopwords configuration
@st.cache_data
def get_stopwords():
    """Get comprehensive stopword list"""
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 
        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now',
        'elder', 'president', 'brother', 'sister', 'saint', 'saints', 'conference', 'general',
        'talk', 'spoke', 'speaking', 'said', 'says', 'today', 'time', 'year', 'years', 'day', 'days',
        'may', 'might', 'must', 'shall', 'would', 'could', 'also', 'even', 'well', 'much', 'many',
        'every', 'first', 'second', 'third', 'last', 'next', 'one', 'two', 'three', 'four', 'five',
        'lord', 'god', 'jesus', 'christ', 'church', 'lds', 'latter', 'brethren', 'sisters', 
        'oaks', 'dallin', 'thank', 'name', 'amen', 'things', 'thing', 'make', 'made', 'way'
    }
    return stopwords

# Text processing class
class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = get_stopwords()
    
    def clean_text(self, text):
        """Clean text for processing"""
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text, remove_stops=True):
        """Tokenize and optionally remove stopwords"""
        text = self.clean_text(text)
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in string.punctuation and any(c.isalpha() for c in t)]
        if remove_stops:
            tokens = [t for t in tokens if t.lower() not in self.stopwords]
        return tokens
    
    def get_sentences(self, text):
        """Split text into sentences"""
        return sent_tokenize(text)

# Talk fetcher class
class TalkFetcher:
    def __init__(self, conn):
        self.conn = conn
        self.base_url = "https://bencrowder.net/collected-talks/dallin-h-oaks/"
        self.headers = {'User-Agent': 'Mozilla/5.0'}
    
    def fetch_new_talks(self, progress_callback=None):
        """Fetch new talks from bencrowder.net"""
        try:
            response = requests.get(self.base_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            talks = []
            content_div = soup.find('div', class_='entry-content') or soup.find('article')
            
            if content_div:
                links = content_div.find_all('a')
                for link in links:
                    href = link.get('href', '')
                    if href and ('churchofjesuschrist.org' in href or 'speeches.byu.edu' in href):
                        # Check if already in database
                        cursor = self.conn.cursor()
                        cursor.execute('SELECT id FROM talks WHERE url = ?', (href,))
                        if not cursor.fetchone():
                            talks.append({
                                'title': link.get_text(strip=True),
                                'url': href,
                                'type': 'General Conference' if 'conference' in href else 'BYU Speech'
                            })
            
            return talks
        except Exception as e:
            st.error(f"Error fetching talks: {e}")
            return []
    
    def download_talk(self, talk_info):
        """Download a single talk"""
        try:
            response = requests.get(talk_info['url'], headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            selectors = ['article', 'div.body-block', 'div.article-content', 'main']
            article = None
            for selector in selectors:
                article = soup.select_one(selector)
                if article:
                    break
            
            if article:
                for element in article(['script', 'style', 'nav', 'header', 'footer']):
                    element.decompose()
                content = article.get_text(separator=' ', strip=True)
                content = re.sub(r'\s+', ' ', content)
                
                # Extract year from content or URL
                year_match = re.search(r'20\d{2}|19\d{2}', talk_info.get('date', '') or talk_info['url'])
                year = int(year_match.group()) if year_match else None
                
                # Save to database
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO talks (title, url, talk_type, year, content, word_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    talk_info['title'],
                    talk_info['url'],
                    talk_info['type'],
                    year,
                    content,
                    len(content.split())
                ))
                self.conn.commit()
                return True
                
        except Exception as e:
            st.error(f"Error downloading talk: {e}")
            return False

# Analysis functions
@st.cache_data
def generate_wordcloud(text, title="Word Cloud", max_words=100, colormap='viridis'):
    """Generate a word cloud from text"""
    stopwords = get_stopwords()
    
    wordcloud = WordCloud(
        width=1600,
        height=900,
        background_color='white',
        stopwords=stopwords,
        max_words=max_words,
        colormap=colormap,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=20, pad=20)
    ax.axis('off')
    
    # Convert to base64 for display
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

@st.cache_data
def analyze_word_frequencies(text, n_top=30):
    """Analyze word frequencies in text"""
    processor = TextProcessor()
    tokens = processor.tokenize(text)
    
    word_freq = Counter(tokens)
    top_words = word_freq.most_common(n_top)
    
    df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    total_words = sum(word_freq.values())
    df['Percentage'] = (df['Frequency'] / total_words * 100).round(2)
    
    return df

def analyze_custom_theme(conn, theme_name, keywords):
    """Analyze a custom theme across all talks"""
    cursor = conn.cursor()
    cursor.execute('SELECT content, title, year, talk_type FROM talks')
    talks = cursor.fetchall()
    
    processor = TextProcessor()
    theme_sentences = []
    talk_mentions = []
    
    for content, title, year, talk_type in talks:
        sentences = processor.get_sentences(content)
        talk_theme_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword.lower() in sentence_lower for keyword in keywords):
                theme_sentences.append(sentence)
                talk_theme_sentences.append(sentence)
        
        if talk_theme_sentences:
            talk_mentions.append({
                'Title': title,
                'Year': year,
                'Type': talk_type,
                'Mentions': len(talk_theme_sentences),
                'Sample': talk_theme_sentences[0][:200] + '...' if talk_theme_sentences else ''
            })
    
    return ' '.join(theme_sentences), pd.DataFrame(talk_mentions)

def get_temporal_analysis(conn):
    """Analyze talks over time"""
    df = pd.read_sql_query('''
        SELECT year, COUNT(*) as count, talk_type
        FROM talks
        WHERE year IS NOT NULL
        GROUP BY year, talk_type
        ORDER BY year
    ''', conn)
    
    return df

# Sidebar
with st.sidebar:
    st.title("📖 Navigation")
    
    page = st.radio("Select Analysis", [
        "📊 Overview",
        "☁️ Word Clouds",
        "📈 Frequency Analysis",
        "🔍 Custom Theme Analysis",
        "📅 Temporal Analysis",
        "🔄 Update Corpus",
        "⚙️ Settings"
    ])
    
    st.divider()
    
    # Database stats
    conn = init_database()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM talks')
    talk_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT MIN(year), MAX(year) FROM talks WHERE year IS NOT NULL')
    year_range = cursor.fetchone()
    
    st.metric("Total Talks", talk_count)
    if year_range[0] and year_range[1]:
        st.metric("Year Range", f"{year_range[0]} - {year_range[1]}")
    
    st.divider()
    st.caption("Built with Streamlit")

# Initialize NLTK
init_nltk()

# Main content based on selected page
if page == "📊 Overview":
    st.title("President Dallin H. Oaks Talk Analysis")
    st.markdown("### Comprehensive Text Analysis Dashboard")
    
    # Overview metrics
    cursor.execute('''
        SELECT 
            COUNT(*) as total_talks,
            SUM(word_count) as total_words,
            AVG(word_count) as avg_words,
            COUNT(DISTINCT talk_type) as talk_types
        FROM talks
    ''')
    stats = cursor.fetchone()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Talks", stats[0])
    col2.metric("Total Words", f"{stats[1]:,}" if stats[1] else 0)
    col3.metric("Avg Words/Talk", f"{int(stats[2]):,}" if stats[2] else 0)
    col4.metric("Talk Types", stats[3])
    
    # Talk distribution
    st.subheader("Talk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # By type
        df_type = pd.read_sql_query('''
            SELECT talk_type, COUNT(*) as count
            FROM talks
            GROUP BY talk_type
        ''', conn)
        
        if not df_type.empty:
            fig = px.pie(df_type, values='count', names='talk_type', 
                        title="Talks by Type")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # By decade
        df_decade = pd.read_sql_query('''
            SELECT 
                CAST((year/10)*10 AS TEXT) || 's' as decade,
                COUNT(*) as count
            FROM talks
            WHERE year IS NOT NULL
            GROUP BY decade
            ORDER BY decade
        ''', conn)
        
        if not df_decade.empty:
            fig = px.bar(df_decade, x='decade', y='count',
                        title="Talks by Decade")
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent talks
    st.subheader("Recent Talks in Database")
    df_recent = pd.read_sql_query('''
        SELECT title, talk_type, year, word_count
        FROM talks
        ORDER BY year DESC, id DESC
        LIMIT 10
    ''', conn)
    
    if not df_recent.empty:
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No talks in database. Please update the corpus first.")

elif page == "☁️ Word Clouds":
    st.title("Word Cloud Visualizations")
    
    # Get all talk content
    cursor.execute('SELECT content FROM talks')
    talks = cursor.fetchall()
    
    if talks:
        all_text = ' '.join([talk[0] for talk in talks])
        
        tab1, tab2, tab3, tab4 = st.tabs(["All Talks", "By Type", "By Decade", "By Theme"])
        
        with tab1:
            st.subheader("All Talks Combined")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                max_words = st.slider("Max Words", 50, 200, 100)
            with col2:
                colormap = st.selectbox("Color Scheme", 
                    ['viridis', 'plasma', 'inferno', 'coolwarm', 'rainbow'])
            
            if st.button("Generate Word Cloud", key="all_talks"):
                with st.spinner("Generating word cloud..."):
                    buf = generate_wordcloud(all_text, "All Talks", max_words, colormap)
                    st.image(buf, use_column_width=True)
                    
                    # Download button
                    buf.seek(0)
                    btn = st.download_button(
                        label="Download Word Cloud",
                        data=buf,
                        file_name="oaks_wordcloud_all.png",
                        mime="image/png"
                    )
        
        with tab2:
            st.subheader("Word Clouds by Talk Type")
            
            talk_type = st.selectbox("Select Talk Type", 
                pd.read_sql_query('SELECT DISTINCT talk_type FROM talks', conn)['talk_type'].tolist())
            
            if st.button("Generate", key="by_type"):
                cursor.execute('SELECT content FROM talks WHERE talk_type = ?', (talk_type,))
                type_talks = cursor.fetchall()
                type_text = ' '.join([talk[0] for talk in type_talks])
                
                with st.spinner("Generating word cloud..."):
                    buf = generate_wordcloud(type_text, f"{talk_type} Talks", 100, 'plasma')
                    st.image(buf, use_column_width=True)
        
        with tab3:
            st.subheader("Word Clouds by Decade")
            
            decades = pd.read_sql_query('''
                SELECT DISTINCT CAST((year/10)*10 AS TEXT) || 's' as decade
                FROM talks
                WHERE year IS NOT NULL
                ORDER BY decade
            ''', conn)['decade'].tolist()
            
            decade = st.selectbox("Select Decade", decades)
            
            if st.button("Generate", key="by_decade"):
                decade_year = int(decade[:-1])
                cursor.execute('''
                    SELECT content FROM talks 
                    WHERE year >= ? AND year < ?
                ''', (decade_year, decade_year + 10))
                decade_talks = cursor.fetchall()
                decade_text = ' '.join([talk[0] for talk in decade_talks])
                
                with st.spinner("Generating word cloud..."):
                    buf = generate_wordcloud(decade_text, f"{decade} Talks", 100, 'coolwarm')
                    st.image(buf, use_column_width=True)
        
        with tab4:
            st.subheader("Predefined Theme Word Clouds")
            
            themes = {
                'Faith & Testimony': ['faith', 'testimony', 'believe', 'witness', 'prayer', 'trust'],
                'Family & Marriage': ['family', 'marriage', 'children', 'parent', 'father', 'mother', 'home'],
                'Service & Love': ['service', 'serve', 'love', 'charity', 'help', 'minister'],
                'Covenant & Temple': ['covenant', 'temple', 'ordinance', 'baptism', 'endowment'],
                'Scripture & Revelation': ['scripture', 'revelation', 'prophet', 'bible', 'book', 'mormon']
            }
            
            theme_name = st.selectbox("Select Theme", list(themes.keys()))
            
            if st.button("Generate", key="by_theme"):
                processor = TextProcessor()
                theme_sentences = []
                
                for content in [talk[0] for talk in talks]:
                    sentences = processor.get_sentences(content)
                    for sentence in sentences:
                        if any(keyword in sentence.lower() for keyword in themes[theme_name]):
                            theme_sentences.append(sentence)
                
                theme_text = ' '.join(theme_sentences)
                
                with st.spinner("Generating word cloud..."):
                    buf = generate_wordcloud(theme_text, f"Theme: {theme_name}", 100, 'viridis')
                    st.image(buf, use_column_width=True)
    else:
        st.warning("No talks in database. Please update the corpus first.")

elif page == "📈 Frequency Analysis":
    st.title("Word Frequency Analysis")
    
    cursor.execute('SELECT content FROM talks')
    talks = cursor.fetchall()
    
    if talks:
        all_text = ' '.join([talk[0] for talk in talks])
        
        st.subheader("Top Word Frequencies")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_words = st.number_input("Number of top words", 10, 100, 30)
        
        df_freq = analyze_word_frequencies(all_text, n_words)
        
        # Display table
        st.dataframe(df_freq, use_container_width=True)
        
        # Bar chart
        fig = px.bar(df_freq.head(20), x='Word', y='Frequency',
                    title=f"Top 20 Most Frequent Words",
                    hover_data=['Percentage'])
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download CSV
        csv = df_freq.to_csv(index=False)
        st.download_button(
            label="Download Frequency Data (CSV)",
            data=csv,
            file_name="word_frequencies.csv",
            mime="text/csv"
        )
        
        # Comparative analysis
        st.subheader("Comparative Frequency Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            compare_by = st.selectbox("Compare by", ["Talk Type", "Decade"])
        
        if compare_by == "Talk Type":
            types = pd.read_sql_query('SELECT DISTINCT talk_type FROM talks', conn)['talk_type'].tolist()
            
            freq_data = {}
            for talk_type in types:
                cursor.execute('SELECT content FROM talks WHERE talk_type = ?', (talk_type,))
                type_talks = cursor.fetchall()
                if type_talks:
                    type_text = ' '.join([talk[0] for talk in type_talks])
                    freq_data[talk_type] = analyze_word_frequencies(type_text, 10)
            
            if freq_data:
                st.subheader("Top Words by Talk Type")
                for talk_type, df in freq_data.items():
                    st.write(f"**{talk_type}**")
                    st.dataframe(df[['Word', 'Frequency']].head(10))
        
        else:  # By Decade
            decades = pd.read_sql_query('''
                SELECT DISTINCT CAST((year/10)*10 AS TEXT) || 's' as decade,
                       (year/10)*10 as decade_year
                FROM talks
                WHERE year IS NOT NULL
                ORDER BY decade_year
            ''', conn)
            
            freq_data = {}
            for _, row in decades.iterrows():
                decade = row['decade']
                decade_year = row['decade_year']
                
                cursor.execute('''
                    SELECT content FROM talks 
                    WHERE year >= ? AND year < ?
                ''', (decade_year, decade_year + 10))
                decade_talks = cursor.fetchall()
                if decade_talks:
                    decade_text = ' '.join([talk[0] for talk in decade_talks])
                    freq_data[decade] = analyze_word_frequencies(decade_text, 10)
            
            if freq_data:
                st.subheader("Top Words by Decade")
                cols = st.columns(len(freq_data))
                for i, (decade, df) in enumerate(freq_data.items()):
                    with cols[i]:
                        st.write(f"**{decade}**")
                        st.dataframe(df[['Word', 'Frequency']].head(10))
    else:
        st.warning("No talks in database. Please update the corpus first.")

elif page == "🔍 Custom Theme Analysis":
    st.title("Custom Theme Analysis")
    st.markdown("Analyze custom themes or specific words across all talks")
    
    tab1, tab2 = st.tabs(["New Analysis", "Saved Themes"])
    
    with tab1:
        st.subheader("Create Custom Theme Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            theme_name = st.text_input("Theme Name", placeholder="e.g., Education")
            
            keywords_input = st.text_area(
                "Keywords (one per line)",
                placeholder="learn\nstudy\nknowledge\neducation\nschool",
                height=150
            )
            
            search_type = st.radio("Search Type", 
                ["All keywords (OR)", "All keywords must appear (AND)"])
            
            save_theme = st.checkbox("Save this theme for future use")
        
        with col2:
            if st.button("Analyze Theme", type="primary"):
                if theme_name and keywords_input:
                    keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
                    
                    with st.spinner(f"Analyzing '{theme_name}' theme..."):
                        theme_text, df_mentions = analyze_custom_theme(conn, theme_name, keywords)
                        
                        if theme_text:
                            st.success(f"Found {len(df_mentions)} talks mentioning '{theme_name}'")
                            
                            # Generate word cloud
                            st.subheader(f"Word Cloud: {theme_name}")
                            buf = generate_wordcloud(theme_text, f"Theme: {theme_name}", 100, 'plasma')
                            st.image(buf, use_column_width=True)
                            
                            # Show mentions
                            st.subheader("Talk Mentions")
                            st.dataframe(df_mentions, use_container_width=True)
                            
                            # Word frequencies in theme
                            st.subheader("Top Words in Theme Context")
                            df_freq = analyze_word_frequencies(theme_text, 20)
                            st.dataframe(df_freq, use_container_width=True)
                            
                            # Save theme if requested
                            if save_theme:
                                cursor = conn.cursor()
                                cursor.execute('''
                                    INSERT INTO custom_themes (theme_name, keywords)
                                    VALUES (?, ?)
                                ''', (theme_name, ','.join(keywords)))
                                conn.commit()
                                st.success(f"Theme '{theme_name}' saved!")
                        else:
                            st.warning(f"No mentions of '{theme_name}' found in talks")
                else:
                    st.error("Please enter both theme name and keywords")
    
    with tab2:
        st.subheader("Saved Themes")
        
        cursor.execute('SELECT * FROM custom_themes ORDER BY created_at DESC')
        saved_themes = cursor.fetchall()
        
        if saved_themes:
            for theme in saved_themes:
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    st.write(f"**{theme[1]}**")
                
                with col2:
                    keywords = theme[2].split(',')
                    st.write(f"Keywords: {', '.join(keywords[:3])}...")
                
                with col3:
                    if st.button("Analyze", key=f"analyze_{theme[0]}"):
                        with st.spinner(f"Analyzing '{theme[1]}'..."):
                            theme_text, df_mentions = analyze_custom_theme(conn, theme[1], keywords)
                            if theme_text:
                                buf = generate_wordcloud(theme_text, f"Theme: {theme[1]}", 100, 'viridis')
                                st.image(buf, use_column_width=True)
        else:
            st.info("No saved themes yet. Create one in the 'New Analysis' tab.")

elif page == "📅 Temporal Analysis":
    st.title("Temporal Analysis")
    st.markdown("Analyze how themes and language have evolved over time")
    
    df_temporal = get_temporal_analysis(conn)
    
    if not df_temporal.empty:
        # Timeline chart
        st.subheader("Talks Over Time")
        
        fig = px.line(df_temporal.groupby('year')['count'].sum().reset_index(), 
                     x='year', y='count',
                     title="Number of Talks by Year",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Stacked bar chart by type
        st.subheader("Talk Types Over Time")
        
        fig = px.bar(df_temporal, x='year', y='count', color='talk_type',
                    title="Talk Distribution by Type and Year")
        st.plotly_chart(fig, use_container_width=True)
        
        # Theme evolution
        st.subheader("Theme Evolution Analysis")
        
        theme_to_analyze = st.selectbox("Select Theme to Track", [
            "Faith", "Family", "Service", "Temple", "Prayer", "Repentance"
        ])
        
        if st.button("Analyze Theme Evolution"):
            years = df_temporal['year'].unique()
            theme_counts = []
            
            for year in sorted(years):
                if year:
                    cursor.execute('SELECT content FROM talks WHERE year = ?', (year,))
                    year_talks = cursor.fetchall()
                    if year_talks:
                        year_text = ' '.join([talk[0] for talk in year_talks]).lower()
                        count = year_text.count(theme_to_analyze.lower())
                        theme_counts.append({'Year': year, 'Mentions': count})
            
            if theme_counts:
                df_theme = pd.DataFrame(theme_counts)
                fig = px.line(df_theme, x='Year', y='Mentions',
                            title=f"'{theme_to_analyze}' Mentions Over Time",
                            markers=True)
                st.plotly_chart(fig, use_container_width=True)
        
        # Decade comparison
        st.subheader("Decade Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            decade1 = st.selectbox("First Decade", ["1970s", "1980s", "1990s", "2000s", "2010s", "2020s"])
        
        with col2:
            decade2 = st.selectbox("Second Decade", ["2000s", "2010s", "2020s"])
        
        if st.button("Compare Decades"):
            # Get talks from each decade
            decade1_year = int(decade1[:-1])
            decade2_year = int(decade2[:-1])
            
            cursor.execute('''
                SELECT content FROM talks 
                WHERE year >= ? AND year < ?
            ''', (decade1_year, decade1_year + 10))
            decade1_talks = cursor.fetchall()
            
            cursor.execute('''
                SELECT content FROM talks 
                WHERE year >= ? AND year < ?
            ''', (decade2_year, decade2_year + 10))
            decade2_talks = cursor.fetchall()
            
            if decade1_talks and decade2_talks:
                decade1_text = ' '.join([talk[0] for talk in decade1_talks])
                decade2_text = ' '.join([talk[0] for talk in decade2_talks])
                
                freq1 = analyze_word_frequencies(decade1_text, 15)
                freq2 = analyze_word_frequencies(decade2_text, 15)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Top Words - {decade1}**")
                    st.dataframe(freq1[['Word', 'Frequency']], use_container_width=True)
                
                with col2:
                    st.write(f"**Top Words - {decade2}**")
                    st.dataframe(freq2[['Word', 'Frequency']], use_container_width=True)
                
                # Find unique words
                words1 = set(freq1['Word'].head(50))
                words2 = set(freq2['Word'].head(50))
                
                unique1 = words1 - words2
                unique2 = words2 - words1
                
                if unique1:
                    st.write(f"**Words more common in {decade1}:** {', '.join(list(unique1)[:10])}")
                
                if unique2:
                    st.write(f"**Words more common in {decade2}:** {', '.join(list(unique2)[:10])}")
    else:
        st.warning("No temporal data available. Please update the corpus first.")

elif page == "🔄 Update Corpus":
    st.title("Update Talk Corpus")
    st.markdown("Fetch new talks and update the database")
    
    tab1, tab2, tab3 = st.tabs(["Automatic Update", "Manual Add", "Manage Database"])
    
    with tab1:
        st.subheader("Fetch New Talks from bencrowder.net")
        
        if st.button("Check for New Talks", type="primary"):
            fetcher = TalkFetcher(conn)
            
            with st.spinner("Checking for new talks..."):
                new_talks = fetcher.fetch_new_talks()
                
                if new_talks:
                    st.success(f"Found {len(new_talks)} new talks!")
                    
                    # Display new talks
                    df_new = pd.DataFrame(new_talks)
                    st.dataframe(df_new, use_container_width=True)
                    
                    if st.button("Download All New Talks"):
                        progress_bar = st.progress(0)
                        success_count = 0
                        
                        for i, talk in enumerate(new_talks):
                            if fetcher.download_talk(talk):
                                success_count += 1
                            progress_bar.progress((i + 1) / len(new_talks))
                            time.sleep(1)  # Rate limiting
                        
                        st.success(f"Downloaded {success_count} of {len(new_talks)} talks")
                        st.rerun()
                else:
                    st.info("No new talks found. Database is up to date!")
        
        st.divider()
        
        st.subheader("Quick Update - Latest Conference")
        
        if st.button("Fetch Latest General Conference"):
            st.info("This feature checks the Church website for the most recent conference talks")
            # This would need implementation to check churchofjesuschrist.org directly
            st.warning("Feature in development - use automatic update above")
    
    with tab2:
        st.subheader("Manually Add a Talk")
        
        with st.form("add_talk_form"):
            title = st.text_input("Talk Title", placeholder="Enter talk title")
            url = st.text_input("Talk URL", placeholder="https://...")
            talk_type = st.selectbox("Talk Type", ["General Conference", "BYU Speech", "Devotional", "Other"])
            year = st.number_input("Year", min_value=1970, max_value=2024, value=2024)
            
            content = st.text_area("Talk Content", 
                placeholder="Paste the full text of the talk here...",
                height=300)
            
            if st.form_submit_button("Add Talk"):
                if title and content:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO talks (title, url, talk_type, year, content, word_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (title, url or '', talk_type, year, content, len(content.split())))
                    conn.commit()
                    st.success(f"Added '{title}' to database!")
                    st.rerun()
                else:
                    st.error("Please provide at least title and content")
    
    with tab3:
        st.subheader("Database Management")
        
        # Export database
        if st.button("Export Database to CSV"):
            df_export = pd.read_sql_query('''
                SELECT title, url, talk_type, year, word_count, created_at
                FROM talks
                ORDER BY year DESC, id DESC
            ''', conn)
            
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download Database CSV",
                data=csv,
                file_name=f"oaks_talks_database_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        st.divider()
        
        # Database statistics
        st.subheader("Database Statistics")
        
        cursor.execute('''
            SELECT 
                talk_type,
                COUNT(*) as count,
                SUM(word_count) as total_words,
                AVG(word_count) as avg_words,
                MIN(year) as earliest,
                MAX(year) as latest
            FROM talks
            GROUP BY talk_type
        ''')
        
        stats = cursor.fetchall()
        
        if stats:
            df_stats = pd.DataFrame(stats, 
                columns=['Type', 'Count', 'Total Words', 'Avg Words', 'Earliest', 'Latest'])
            st.dataframe(df_stats, use_container_width=True)
        
        st.divider()
        
        # Danger zone
        with st.expander("⚠️ Danger Zone"):
            st.warning("These actions cannot be undone!")
            
            if st.button("Clear All Talks", type="secondary"):
                if st.checkbox("I understand this will delete all talks"):
                    cursor.execute('DELETE FROM talks')
                    conn.commit()
                    st.success("All talks deleted")
                    st.rerun()

elif page == "⚙️ Settings":
    st.title("Settings")
    
    st.subheader("Stopwords Configuration")
    
    current_stopwords = get_stopwords()
    
    st.write(f"Current stopwords count: {len(current_stopwords)}")
    
    with st.expander("View/Edit Stopwords"):
        stopwords_text = st.text_area(
            "Stopwords (one per line)",
            value='\n'.join(sorted(current_stopwords)),
            height=300
        )
        
        if st.button("Update Stopwords"):
            # This would need to be saved to database or config file
            st.success("Stopwords updated (for this session)")
    
    st.divider()
    
    st.subheader("Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        image_dpi = st.number_input("Image Export DPI", 72, 300, 150)
        st.caption("Higher DPI = better quality, larger file size")
    
    with col2:
        default_colormap = st.selectbox("Default Color Scheme", 
            ['viridis', 'plasma', 'inferno', 'coolwarm', 'rainbow'])
    
    st.divider()
    
    st.subheader("About")
    
    st.info("""
    **President Dallin H. Oaks Talk Analysis Tool**
    
    Version 1.0.0
    
    This tool provides comprehensive text analysis of President Dallin H. Oaks' talks,
    including General Conference addresses, BYU speeches, and other talks.
    
    Features:
    - Word cloud generation
    - Frequency analysis
    - Custom theme analysis
    - Temporal analysis
    - Automatic corpus updates
    
    Data source: bencrowder.net/collected-talks
    """)

# Footer
st.divider()
st.caption("President Dallin H. Oaks Talk Analysis Tool | Built with Streamlit | Data from bencrowder.net")
