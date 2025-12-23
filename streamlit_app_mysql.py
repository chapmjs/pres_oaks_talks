"""
President Dallin H. Oaks Talk Analysis - Streamlit App (MySQL Version)
Enterprise-ready version with MySQL backend for better scalability
"""

import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import pooling, Error
import os
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
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
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="President Oaks Talk Analysis",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database configuration from environment variables or defaults
DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'port': int(os.getenv('MYSQL_PORT', 3306)),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', 'password'),
    'database': os.getenv('MYSQL_DATABASE', 'oaks_talks'),
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci',
    'use_unicode': True,
    'autocommit': True,
    'pool_name': 'oaks_pool',
    'pool_size': 5,
    'pool_reset_session': True
}

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

# Create connection pool
@st.cache_resource
def init_connection_pool():
    """Initialize MySQL connection pool"""
    try:
        # First, try to connect without specifying database to create it if needed
        temp_config = DB_CONFIG.copy()
        db_name = temp_config.pop('database')
        temp_config.pop('pool_name', None)
        temp_config.pop('pool_size', None)
        temp_config.pop('pool_reset_session', None)
        
        # Create database if it doesn't exist
        try:
            temp_conn = mysql.connector.connect(**temp_config)
            cursor = temp_conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.close()
            temp_conn.close()
            logger.info(f"Database '{db_name}' verified/created")
        except Error as e:
            logger.error(f"Error creating database: {e}")
        
        # Now create the connection pool
        pool = pooling.MySQLConnectionPool(**DB_CONFIG)
        logger.info("MySQL connection pool created successfully")
        return pool
    except Error as e:
        logger.error(f"Error creating connection pool: {e}")
        st.error(f"Database connection failed: {e}")
        return None

# Initialize database schema
def init_database_schema(pool):
    """Create database tables if they don't exist"""
    if not pool:
        return False
    
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        
        # Create talks table with proper indexes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS talks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(500) NOT NULL,
                url VARCHAR(512) UNIQUE,
                talk_type VARCHAR(100),
                date VARCHAR(100),
                year INT,
                content LONGTEXT,
                word_count INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_year (year),
                INDEX idx_type (talk_type),
                INDEX idx_url (url),
                FULLTEXT idx_content (content)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Create custom themes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS custom_themes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                theme_name VARCHAR(200) NOT NULL,
                keywords TEXT,
                created_by VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_theme_name (theme_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Create analysis cache table for performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_cache (
                id INT AUTO_INCREMENT PRIMARY KEY,
                cache_key VARCHAR(255) UNIQUE,
                cache_type VARCHAR(50),
                cache_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                INDEX idx_cache_key (cache_key),
                INDEX idx_cache_type (cache_type),
                INDEX idx_expires (expires_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Create user sessions table for tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100) UNIQUE,
                user_ip VARCHAR(45),
                actions JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_session (session_id),
                INDEX idx_last_active (last_active)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database schema initialized successfully")
        return True
        
    except Error as e:
        logger.error(f"Error initializing database schema: {e}")
        st.error(f"Failed to initialize database: {e}")
        return False

# Get database connection from pool
def get_db_connection(pool):
    """Get a connection from the pool"""
    try:
        return pool.get_connection()
    except Error as e:
        logger.error(f"Error getting connection from pool: {e}")
        return None

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

# Cache management
def get_cached_analysis(pool, cache_key, cache_type='general'):
    """Retrieve cached analysis results"""
    conn = get_db_connection(pool)
    if not conn:
        return None
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('''
            SELECT cache_data FROM analysis_cache 
            WHERE cache_key = %s AND cache_type = %s 
            AND (expires_at IS NULL OR expires_at > NOW())
        ''', (cache_key, cache_type))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            return json.loads(result['cache_data'])
        return None
        
    except Error as e:
        logger.error(f"Error retrieving cache: {e}")
        if conn:
            conn.close()
        return None

def set_cached_analysis(pool, cache_key, cache_data, cache_type='general', ttl_hours=24):
    """Store analysis results in cache"""
    conn = get_db_connection(pool)
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        cursor.execute('''
            INSERT INTO analysis_cache (cache_key, cache_type, cache_data, expires_at)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
            cache_data = VALUES(cache_data),
            expires_at = VALUES(expires_at)
        ''', (cache_key, cache_type, json.dumps(cache_data), expires_at))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Error as e:
        logger.error(f"Error setting cache: {e}")
        if conn:
            conn.close()
        return False

# Talk fetcher class
class TalkFetcher:
    def __init__(self, pool):
        self.pool = pool
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
                        if not self.talk_exists(href):
                            talks.append({
                                'title': link.get_text(strip=True),
                                'url': href,
                                'type': 'General Conference' if 'conference' in href else 'BYU Speech'
                            })
            
            return talks
        except Exception as e:
            logger.error(f"Error fetching talks: {e}")
            return []
    
    def talk_exists(self, url):
        """Check if talk already exists in database"""
        conn = get_db_connection(self.pool)
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM talks WHERE url = %s', (url,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return result is not None
        except Error as e:
            logger.error(f"Error checking talk existence: {e}")
            if conn:
                conn.close()
            return False
    
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
                
                # Extract year
                year_match = re.search(r'20\d{2}|19\d{2}', talk_info.get('date', '') or talk_info['url'])
                year = int(year_match.group()) if year_match else None
                
                # Save to database
                conn = get_db_connection(self.pool)
                if conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO talks (title, url, talk_type, year, content, word_count)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        content = VALUES(content),
                        word_count = VALUES(word_count)
                    ''', (
                        talk_info['title'],
                        talk_info['url'],
                        talk_info['type'],
                        year,
                        content,
                        len(content.split())
                    ))
                    conn.commit()
                    cursor.close()
                    conn.close()
                    return True
                
        except Exception as e:
            logger.error(f"Error downloading talk: {e}")
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

def analyze_custom_theme(pool, theme_name, keywords):
    """Analyze a custom theme across all talks"""
    conn = get_db_connection(pool)
    if not conn:
        return '', pd.DataFrame()
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT content, title, year, talk_type FROM talks')
        talks = cursor.fetchall()
        cursor.close()
        conn.close()
        
        processor = TextProcessor()
        theme_sentences = []
        talk_mentions = []
        
        for talk in talks:
            sentences = processor.get_sentences(talk['content'])
            talk_theme_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword.lower() in sentence_lower for keyword in keywords):
                    theme_sentences.append(sentence)
                    talk_theme_sentences.append(sentence)
            
            if talk_theme_sentences:
                talk_mentions.append({
                    'Title': talk['title'],
                    'Year': talk['year'],
                    'Type': talk['talk_type'],
                    'Mentions': len(talk_theme_sentences),
                    'Sample': talk_theme_sentences[0][:200] + '...' if talk_theme_sentences else ''
                })
        
        return ' '.join(theme_sentences), pd.DataFrame(talk_mentions)
        
    except Error as e:
        logger.error(f"Error analyzing theme: {e}")
        if conn:
            conn.close()
        return '', pd.DataFrame()

def get_temporal_analysis(pool):
    """Analyze talks over time"""
    conn = get_db_connection(pool)
    if not conn:
        return pd.DataFrame()
    
    try:
        query = '''
            SELECT year, COUNT(*) as count, talk_type
            FROM talks
            WHERE year IS NOT NULL
            GROUP BY year, talk_type
            ORDER BY year
        '''
        df = pd.read_sql(query, conn)
        conn.close()
        return df
        
    except Error as e:
        logger.error(f"Error in temporal analysis: {e}")
        if conn:
            conn.close()
        return pd.DataFrame()

# Initialize resources
init_nltk()
pool = init_connection_pool()

if pool:
    init_database_schema(pool)

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
    if pool:
        conn = get_db_connection(pool)
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM talks')
                talk_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT MIN(year), MAX(year) FROM talks WHERE year IS NOT NULL')
                year_range = cursor.fetchone()
                
                cursor.close()
                conn.close()
                
                st.metric("Total Talks", talk_count)
                if year_range[0] and year_range[1]:
                    st.metric("Year Range", f"{year_range[0]} - {year_range[1]}")
            except:
                st.warning("Database not initialized")
    else:
        st.error("Database connection failed")
        st.stop()
    
    st.divider()
    
    # MySQL connection info
    with st.expander("Database Info"):
        st.code(f"""
Host: {DB_CONFIG['host']}
Port: {DB_CONFIG['port']}
Database: {DB_CONFIG['database']}
Pool Size: {DB_CONFIG['pool_size']}
        """)
    
    st.caption("MySQL-Powered Analysis")

# Main content based on selected page
if page == "📊 Overview":
    st.title("President Dallin H. Oaks Talk Analysis")
    st.markdown("### MySQL-Powered Comprehensive Analysis Dashboard")
    
    if pool:
        conn = get_db_connection(pool)
        if conn:
            try:
                cursor = conn.cursor()
                
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
                col1.metric("Total Talks", stats[0] if stats[0] else 0)
                col2.metric("Total Words", f"{stats[1]:,}" if stats[1] else 0)
                col3.metric("Avg Words/Talk", f"{int(stats[2]):,}" if stats[2] else 0)
                col4.metric("Talk Types", stats[3] if stats[3] else 0)
                
                # Talk distribution
                st.subheader("Talk Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # By type
                    cursor.execute('''
                        SELECT talk_type, COUNT(*) as count
                        FROM talks
                        GROUP BY talk_type
                    ''')
                    type_data = cursor.fetchall()
                    
                    if type_data:
                        df_type = pd.DataFrame(type_data, columns=['talk_type', 'count'])
                        fig = px.pie(df_type, values='count', names='talk_type', 
                                    title="Talks by Type")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # By decade
                    cursor.execute('''
                        SELECT 
                            CONCAT(FLOOR(year/10)*10, 's') as decade,
                            COUNT(*) as count
                        FROM talks
                        WHERE year IS NOT NULL
                        GROUP BY decade
                        ORDER BY decade
                    ''')
                    decade_data = cursor.fetchall()
                    
                    if decade_data:
                        df_decade = pd.DataFrame(decade_data, columns=['decade', 'count'])
                        fig = px.bar(df_decade, x='decade', y='count',
                                    title="Talks by Decade")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Recent talks
                st.subheader("Recent Talks in Database")
                cursor.execute('''
                    SELECT title, talk_type, year, word_count
                    FROM talks
                    ORDER BY year DESC, id DESC
                    LIMIT 10
                ''')
                recent_talks = cursor.fetchall()
                
                if recent_talks:
                    df_recent = pd.DataFrame(recent_talks, 
                        columns=['Title', 'Type', 'Year', 'Word Count'])
                    st.dataframe(df_recent, use_container_width=True, hide_index=True)
                else:
                    st.info("No talks in database. Please update the corpus first.")
                
                cursor.close()
                conn.close()
                
            except Error as e:
                st.error(f"Database error: {e}")
                if conn:
                    conn.close()

elif page == "☁️ Word Clouds":
    st.title("Word Cloud Visualizations")
    
    if pool:
        conn = get_db_connection(pool)
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT content FROM talks')
                talks = cursor.fetchall()
                cursor.close()
                conn.close()
                
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
                                
                                buf.seek(0)
                                st.download_button(
                                    label="Download Word Cloud",
                                    data=buf,
                                    file_name="oaks_wordcloud_all.png",
                                    mime="image/png"
                                )
                    
                    with tab2:
                        st.subheader("Word Clouds by Talk Type")
                        
                        conn = get_db_connection(pool)
                        cursor = conn.cursor()
                        cursor.execute('SELECT DISTINCT talk_type FROM talks')
                        types = [t[0] for t in cursor.fetchall()]
                        cursor.close()
                        conn.close()
                        
                        talk_type = st.selectbox("Select Talk Type", types)
                        
                        if st.button("Generate", key="by_type"):
                            conn = get_db_connection(pool)
                            cursor = conn.cursor()
                            cursor.execute('SELECT content FROM talks WHERE talk_type = %s', (talk_type,))
                            type_talks = cursor.fetchall()
                            cursor.close()
                            conn.close()
                            
                            type_text = ' '.join([talk[0] for talk in type_talks])
                            
                            with st.spinner("Generating word cloud..."):
                                buf = generate_wordcloud(type_text, f"{talk_type} Talks", 100, 'plasma')
                                st.image(buf, use_column_width=True)
                    
                    # Continue with other tabs...
                else:
                    st.warning("No talks in database. Please update the corpus first.")
                    
            except Error as e:
                st.error(f"Database error: {e}")

elif page == "📈 Frequency Analysis":
    st.title("Word Frequency Analysis")
    
    if pool:
        conn = get_db_connection(pool)
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT content FROM talks')
                talks = cursor.fetchall()
                cursor.close()
                conn.close()
                
                if talks:
                    all_text = ' '.join([talk[0] for talk in talks])
                    
                    st.subheader("Top Word Frequencies")
                    
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        n_words = st.number_input("Number of top words", 10, 100, 30)
                    
                    # Check cache first
                    cache_key = f"freq_{n_words}"
                    cached = get_cached_analysis(pool, cache_key, 'frequency')
                    
                    if cached:
                        df_freq = pd.DataFrame(cached)
                    else:
                        df_freq = analyze_word_frequencies(all_text, n_words)
                        set_cached_analysis(pool, cache_key, df_freq.to_dict('records'), 'frequency', 4)
                    
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
                else:
                    st.warning("No talks in database. Please update the corpus first.")
                    
            except Error as e:
                st.error(f"Database error: {e}")

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
            
            save_theme = st.checkbox("Save this theme for future use")
        
        with col2:
            if st.button("Analyze Theme", type="primary"):
                if theme_name and keywords_input:
                    keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
                    
                    with st.spinner(f"Analyzing '{theme_name}' theme..."):
                        theme_text, df_mentions = analyze_custom_theme(pool, theme_name, keywords)
                        
                        if theme_text:
                            st.success(f"Found {len(df_mentions)} talks mentioning '{theme_name}'")
                            
                            # Generate word cloud
                            st.subheader(f"Word Cloud: {theme_name}")
                            buf = generate_wordcloud(theme_text, f"Theme: {theme_name}", 100, 'plasma')
                            st.image(buf, use_column_width=True)
                            
                            # Show mentions
                            st.subheader("Talk Mentions")
                            st.dataframe(df_mentions, use_container_width=True)
                            
                            # Save theme if requested
                            if save_theme and pool:
                                conn = get_db_connection(pool)
                                if conn:
                                    cursor = conn.cursor()
                                    cursor.execute('''
                                        INSERT INTO custom_themes (theme_name, keywords)
                                        VALUES (%s, %s)
                                    ''', (theme_name, ','.join(keywords)))
                                    conn.commit()
                                    cursor.close()
                                    conn.close()
                                    st.success(f"Theme '{theme_name}' saved!")
                        else:
                            st.warning(f"No mentions of '{theme_name}' found in talks")
                else:
                    st.error("Please enter both theme name and keywords")
    
    with tab2:
        st.subheader("Saved Themes")
        
        if pool:
            conn = get_db_connection(pool)
            if conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute('SELECT * FROM custom_themes ORDER BY created_at DESC')
                saved_themes = cursor.fetchall()
                cursor.close()
                conn.close()
                
                if saved_themes:
                    for theme in saved_themes:
                        col1, col2, col3 = st.columns([2, 3, 1])
                        
                        with col1:
                            st.write(f"**{theme['theme_name']}**")
                        
                        with col2:
                            keywords = theme['keywords'].split(',')
                            st.write(f"Keywords: {', '.join(keywords[:3])}...")
                        
                        with col3:
                            if st.button("Analyze", key=f"analyze_{theme['id']}"):
                                with st.spinner(f"Analyzing '{theme['theme_name']}'..."):
                                    theme_text, df_mentions = analyze_custom_theme(
                                        pool, theme['theme_name'], keywords)
                                    if theme_text:
                                        buf = generate_wordcloud(
                                            theme_text, f"Theme: {theme['theme_name']}", 100, 'viridis')
                                        st.image(buf, use_column_width=True)
                else:
                    st.info("No saved themes yet. Create one in the 'New Analysis' tab.")

elif page == "📅 Temporal Analysis":
    st.title("Temporal Analysis")
    st.markdown("Analyze how themes and language have evolved over time")
    
    df_temporal = get_temporal_analysis(pool)
    
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
            
            conn = get_db_connection(pool)
            if conn:
                cursor = conn.cursor()
                
                for year in sorted(years):
                    if year:
                        cursor.execute('SELECT content FROM talks WHERE year = %s', (year,))
                        year_talks = cursor.fetchall()
                        if year_talks:
                            year_text = ' '.join([talk[0] for talk in year_talks]).lower()
                            count = year_text.count(theme_to_analyze.lower())
                            theme_counts.append({'Year': year, 'Mentions': count})
                
                cursor.close()
                conn.close()
                
                if theme_counts:
                    df_theme = pd.DataFrame(theme_counts)
                    fig = px.line(df_theme, x='Year', y='Mentions',
                                title=f"'{theme_to_analyze}' Mentions Over Time",
                                markers=True)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No temporal data available. Please update the corpus first.")

elif page == "🔄 Update Corpus":
    st.title("Update Talk Corpus")
    st.markdown("Fetch new talks and update the MySQL database")
    
    tab1, tab2, tab3 = st.tabs(["Automatic Update", "Manual Add", "Database Management"])
    
    with tab1:
        st.subheader("Fetch New Talks from bencrowder.net")
        
        if st.button("Check for New Talks", type="primary"):
            st.write(new_talks)
            fetcher = TalkFetcher(pool)
            
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
                        
                        # Clear cache after update
                        if pool:
                            conn = get_db_connection(pool)
                            if conn:
                                cursor = conn.cursor()
                                cursor.execute('DELETE FROM analysis_cache WHERE expires_at < NOW()')
                                conn.commit()
                                cursor.close()
                                conn.close()
                        
                        st.rerun()
                else:
                    st.info("No new talks found. Database is up to date!")
    
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
                if title and content and pool:
                    conn = get_db_connection(pool)
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO talks (title, url, talk_type, year, content, word_count)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        ''', (title, url or '', talk_type, year, content, len(content.split())))
                        conn.commit()
                        cursor.close()
                        conn.close()
                        st.success(f"Added '{title}' to database!")
                        st.rerun()
                else:
                    st.error("Please provide at least title and content")
    
    with tab3:
        st.subheader("Database Management")
        
        # Export database
        if st.button("Export Database to CSV"):
            if pool:
                conn = get_db_connection(pool)
                df_export = pd.read_sql('''
                    SELECT title, url, talk_type, year, word_count, created_at
                    FROM talks
                    ORDER BY year DESC, id DESC
                ''', conn)
                conn.close()
                
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
        
        if pool:
            conn = get_db_connection(pool)
            if conn:
                cursor = conn.cursor(dictionary=True)
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
                cursor.close()
                conn.close()
                
                if stats:
                    df_stats = pd.DataFrame(stats)
                    st.dataframe(df_stats, use_container_width=True)
        
        st.divider()
        
        # Cache management
        st.subheader("Cache Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Analysis Cache"):
                if pool:
                    conn = get_db_connection(pool)
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute('TRUNCATE TABLE analysis_cache')
                        conn.commit()
                        cursor.close()
                        conn.close()
                        st.success("Cache cleared!")
        
        with col2:
            if pool:
                conn = get_db_connection(pool)
                if conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM analysis_cache')
                    cache_count = cursor.fetchone()[0]
                    cursor.close()
                    conn.close()
                    st.metric("Cached Analyses", cache_count)

elif page == "⚙️ Settings":
    st.title("Settings")
    
    st.subheader("Database Configuration")
    
    # Show current configuration
    st.info(f"""
    **Current MySQL Configuration:**
    - Host: `{DB_CONFIG['host']}`
    - Port: `{DB_CONFIG['port']}`
    - Database: `{DB_CONFIG['database']}`
    - User: `{DB_CONFIG['user']}`
    - Pool Size: `{DB_CONFIG['pool_size']}`
    """)
    
    st.subheader("Environment Variables")
    
    st.markdown("""
    To configure the database connection, set these environment variables:
    
    ```bash
    export MYSQL_HOST=localhost
    export MYSQL_PORT=3306
    export MYSQL_USER=your_user
    export MYSQL_PASSWORD=your_password
    export MYSQL_DATABASE=oaks_talks
    ```
    
    Or create a `.env` file in the app directory:
    
    ```
    MYSQL_HOST=localhost
    MYSQL_PORT=3306
    MYSQL_USER=your_user
    MYSQL_PASSWORD=your_password
    MYSQL_DATABASE=oaks_talks
    ```
    """)
    
    st.divider()
    
    st.subheader("Database Optimization")
    
    if st.button("Optimize Tables"):
        if pool:
            conn = get_db_connection(pool)
            if conn:
                cursor = conn.cursor()
                cursor.execute('OPTIMIZE TABLE talks')
                cursor.execute('OPTIMIZE TABLE custom_themes')
                cursor.execute('OPTIMIZE TABLE analysis_cache')
                conn.commit()
                cursor.close()
                conn.close()
                st.success("Tables optimized!")
    
    st.divider()
    
    st.subheader("About")
    
    st.info("""
    **President Dallin H. Oaks Talk Analysis Tool**
    
    MySQL Version 2.0.0
    
    This enterprise-ready version uses MySQL for:
    - Better concurrent user support
    - Scalable cloud deployment
    - Full-text search capabilities
    - Robust caching system
    - Connection pooling
    
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
st.caption("President Dallin H. Oaks Talk Analysis Tool | MySQL-Powered | Built with Streamlit")
