# President Dallin H. Oaks Talk Analysis - Streamlit App

A comprehensive web application for analyzing President Dallin H. Oaks' talks with advanced text analysis features, custom theme search, and automatic corpus updates.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)

## 🌟 Features

### 📊 Comprehensive Analysis Tools
- **Word Clouds** - Generate beautiful visualizations by decade, type, or theme
- **Frequency Analysis** - Track top words and phrases with statistical breakdowns
- **Temporal Analysis** - See how language and themes evolved over time
- **Custom Theme Analysis** - Search for any word or concept across all talks

### 🔄 Automatic Updates
- **Corpus Management** - Automatically fetch new talks from bencrowder.net
- **Manual Entry** - Add specific talks manually
- **Database Management** - Export, import, and manage your talk collection

### 🎨 Interactive Visualizations
- Interactive Plotly charts
- Downloadable word clouds
- Comparative analysis views
- Mobile-responsive design

## 🚀 Quick Start

### Local Installation

1. **Clone or download the files:**
```bash
git clone https://github.com/yourusername/oaks-analysis-app.git
cd oaks-analysis-app
```

2. **Install dependencies:**
```bash
pip install -r requirements_streamlit.txt
```

3. **Run the app:**
```bash
streamlit run streamlit_app.py
```

4. **Open in browser:**
Navigate to `http://localhost:8501`

### First Time Setup

1. Go to "🔄 Update Corpus" in the sidebar
2. Click "Check for New Talks"
3. Click "Download All New Talks" to populate the database
4. Start exploring with the analysis tools!

## 📱 App Navigation

### Sidebar Pages

#### 📊 Overview
- Database statistics
- Talk distribution charts
- Recent talks listing
- Quick metrics dashboard

#### ☁️ Word Clouds
- **All Talks** - Combined visualization
- **By Type** - General Conference vs BYU Speeches
- **By Decade** - See evolution over time
- **By Theme** - Pre-defined gospel themes

#### 📈 Frequency Analysis
- Top word frequencies with percentages
- Downloadable CSV reports
- Comparative analysis by type or decade
- Interactive bar charts

#### 🔍 Custom Theme Analysis
- **Create custom searches** - Enter any theme and keywords
- **Save themes** - Store frequently used analyses
- **View mentions** - See which talks contain your themes
- **Generate focused word clouds** - Visualize theme-specific language

#### 📅 Temporal Analysis
- Timeline of talks
- Theme evolution tracking
- Decade comparisons
- Trend identification

#### 🔄 Update Corpus
- **Automatic updates** - Fetch from bencrowder.net
- **Manual entry** - Add individual talks
- **Database management** - Export/import data

#### ⚙️ Settings
- Configure stopwords
- Adjust visualization settings
- Export preferences
- About information

## 💡 Use Cases

### For Educators (BYU-Idaho)
- **Course Integration** - Demonstrate text analysis in SCM courses
- **Student Projects** - Template for analyzing business communications
- **Research** - Track evolution of church leadership messages

### For Personal Study
- **Theme Research** - Find all talks on specific topics
- **Conference Preparation** - Review historical talks by theme
- **Language Patterns** - Understand emphasis changes over time

### For Researchers
- **Linguistic Analysis** - Study language evolution
- **Content Analysis** - Track topic frequency
- **Comparative Studies** - Compare with other speakers (modify code)

## 🎯 Custom Theme Analysis Examples

### Example 1: Education Theme
```
Theme Name: Education
Keywords:
- learn
- study
- knowledge
- education
- school
- university
- student
```

### Example 2: Technology & Modern Challenges
```
Theme Name: Technology
Keywords:
- technology
- internet
- social media
- digital
- online
- computer
- artificial intelligence
```

### Example 3: Family Relationships
```
Theme Name: Family Unity
Keywords:
- family
- together
- unity
- love
- support
- relationship
- bond
```

## 🛠️ Advanced Features

### Database Options

**SQLite (Default)**
- File-based, no setup required
- Good for personal use and small teams
- Included in Python standard library

**PostgreSQL (Scalable)**
- Better for multiple concurrent users
- Cloud-hosted options available
- Modify connection string in code

### Customization

#### Change Stopwords
1. Go to Settings page
2. Click "View/Edit Stopwords"
3. Add or remove words
4. Changes apply to current session

#### Modify Themes
Edit the themes dictionary in the code:
```python
themes = {
    'Your Theme': ['keyword1', 'keyword2', 'keyword3'],
    # Add more themes
}
```

#### Adjust Visualizations
- Change color schemes in word cloud generation
- Modify chart types in Plotly visualizations
- Adjust image resolution for exports

## 📊 Data Management

### Export Data
- **CSV Export** - Download talk metadata
- **Word Frequency Export** - Get analysis results
- **Word Cloud Download** - Save visualizations

### Backup Database
```bash
# Backup SQLite database
cp oaks_talks.db oaks_talks_backup.db

# Restore from backup
cp oaks_talks_backup.db oaks_talks.db
```

## 🐛 Troubleshooting

### Common Issues

**"No talks in database"**
- Go to Update Corpus page
- Click "Check for New Talks"
- Download available talks

**"NLTK data not found"**
```python
import nltk
nltk.download('all')
```

**Word clouds not generating**
- Check if talks are in database
- Verify NLTK data is installed
- Ensure sufficient memory available

**Database locked error**
- Close other connections to database
- Restart the app
- Consider PostgreSQL for production

## 🚀 Deployment

### Quick Deploy to Streamlit Cloud (FREE)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Deploy!

See `STREAMLIT_DEPLOYMENT.md` for detailed deployment instructions.

## 📈 Performance Tips

1. **Use caching** - Already implemented with `@st.cache_data`
2. **Batch operations** - Process multiple talks together
3. **Optimize word clouds** - Reduce resolution for faster generation
4. **Database indexes** - Already created on year and type columns

## 🔒 Privacy & Security

- All data stored locally by default
- No external analytics or tracking
- Optional cloud deployment for sharing
- User data never leaves your control

## 📚 Educational Applications

### For Supply Chain Management Courses

1. **Text Mining Demo** - Show real-world NLP applications
2. **Data Visualization** - Teach dashboard creation
3. **Database Concepts** - Demonstrate SQL in practice
4. **Python Applications** - Beyond Excel automation

### Student Projects Ideas

- Analyze company communications
- Track industry terminology evolution
- Compare leadership messaging styles
- Build custom analysis tools

## 🤝 Contributing

Feel free to fork and modify for your needs:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📜 License

MIT License - Feel free to use and modify

## 🙏 Acknowledgments

- Talk collection from [bencrowder.net](https://bencrowder.net/collected-talks)
- Built with [Streamlit](https://streamlit.io)
- NLP powered by [NLTK](https://www.nltk.org)
- Visualizations by [Plotly](https://plotly.com) and [WordCloud](https://github.com/amueller/word_cloud)

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check Streamlit documentation
- Review the deployment guide

---

**Made with ❤️ for gospel study and educational purposes**

*Perfect for BYU-Idaho Operations Management courses demonstrating real-world Python applications!*
