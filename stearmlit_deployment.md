# Streamlit App Deployment Guide
## President Dallin H. Oaks Talk Analysis Tool

This guide covers multiple deployment options for your Streamlit app, from local testing to cloud deployment.

---

## 🚀 Quick Start (Local)

### 1. Install Requirements
```bash
pip install -r requirements_streamlit.txt
```

### 2. Run the App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## ☁️ Cloud Deployment Options

### Option 1: Streamlit Community Cloud (FREE - Recommended)

**Perfect for:** Quick deployment, sharing with others, no server management

1. **Push to GitHub:**
```bash
git init
git add streamlit_app.py requirements_streamlit.txt
git commit -m "Oaks talk analysis Streamlit app"
git remote add origin https://github.com/yourusername/oaks-analysis-app.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud:**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub
- Click "New app"
- Select your repository
- Set main file path: `streamlit_app.py`
- Click "Deploy"

3. **Your app will be available at:**
`https://yourusername-oaks-analysis-app-streamlit-app-xxxxx.streamlit.app`

### Option 2: Heroku (Hobby Plan ~$7/month)

**Perfect for:** More control, custom domain, larger database

1. **Create Heroku files:**

`Procfile`:
```
web: sh setup.sh && streamlit run streamlit_app.py
```

`setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
```

2. **Deploy to Heroku:**
```bash
heroku create oaks-analysis-app
git add .
git commit -m "Add Heroku config"
git push heroku main
```

### Option 3: Google Cloud Run

**Perfect for:** Scalability, Google integration, pay-per-use

1. **Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY streamlit_app.py .

EXPOSE 8080

CMD streamlit run streamlit_app.py \
    --server.port 8080 \
    --server.address 0.0.0.0
```

2. **Deploy:**
```bash
gcloud run deploy oaks-analysis \
    --source . \
    --region us-central1 \
    --allow-unauthenticated
```

### Option 4: Local Network (Office/School)

**Perfect for:** BYU-Idaho classroom use, no internet required

```bash
# Run with network access enabled
streamlit run streamlit_app.py --server.address 0.0.0.0

# Students can access at: http://your-computer-ip:8501
```

---

## 📊 Database Considerations

### SQLite (Current - Recommended for Start)
- **Pros:** Simple, no setup, file-based
- **Cons:** Limited concurrent users, size limits on free hosting
- **Best for:** Personal use, small teams, demos

### PostgreSQL (Upgrade Option)
- **When to upgrade:** >10 concurrent users, >100MB data
- **Services:** Supabase (free tier), Neon, Railway

**To migrate to PostgreSQL:**

1. Install psycopg2:
```bash
pip install psycopg2-binary
```

2. Update connection in app:
```python
import psycopg2
conn = psycopg2.connect(
    host="your-db-host",
    database="oaks_talks",
    user="username",
    password="password"
)
```

---

## 🎨 Customization

### Change App Appearance

Edit the app configuration:
```python
st.set_page_config(
    page_title="Your Custom Title",
    page_icon="🎓",  # Change icon
    layout="wide",
    theme={
        "primaryColor": "#1e5c2e",  # BYU-I green
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730"
    }
)
```

### Add BYU-Idaho Branding

```python
# Add to sidebar
st.sidebar.image("byui_logo.png", width=200)
st.sidebar.markdown("### BYU-Idaho SCM Analysis Tool")
```

### Restrict Access (for sensitive data)

```python
# Simple password protection
password = st.text_input("Password", type="password")
if password != st.secrets["password"]:
    st.error("Please enter the correct password")
    st.stop()
```

---

## 🔧 Performance Optimization

### 1. Enable Caching
The app already uses `@st.cache_data` for heavy computations. Ensure it's working:
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def expensive_function():
    # Your analysis code
    return results
```

### 2. Optimize Database Queries
- Add indexes on frequently queried columns
- Use connection pooling for PostgreSQL
- Implement pagination for large result sets

### 3. Reduce Word Cloud Generation Time
```python
# Lower resolution for faster generation
wordcloud = WordCloud(
    width=800,  # Instead of 1600
    height=400,  # Instead of 900
    max_words=50  # Instead of 100
)
```

---

## 🐛 Troubleshooting

### "NLTK data not found"
```python
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('all')
```

### "Database locked" (SQLite)
- Reduce concurrent writes
- Implement retry logic
- Consider PostgreSQL for production

### Memory issues on free hosting
- Reduce cache size
- Process talks in batches
- Use smaller word cloud resolutions

### Slow initial load
- Pre-populate database
- Use GitHub LFS for large data files
- Implement lazy loading

---

## 📱 Mobile Optimization

The app is mobile-responsive by default, but you can enhance it:

```python
# Detect mobile
if st.sidebar.button("📱 Mobile View"):
    st.set_page_config(layout="centered")
    
# Adjust visualizations for mobile
if mobile_view:
    fig.update_layout(height=300)  # Smaller charts
    wordcloud_width = 400  # Smaller word clouds
```

---

## 🔐 Security Best Practices

1. **Never commit sensitive data:**
```bash
# .gitignore
*.db
.env
secrets.toml
```

2. **Use environment variables:**
```python
import os
DB_PASSWORD = os.environ.get('DB_PASSWORD')
```

3. **Implement rate limiting:**
```python
from datetime import datetime, timedelta

if 'last_update' in st.session_state:
    if datetime.now() - st.session_state.last_update < timedelta(minutes=5):
        st.error("Please wait 5 minutes between updates")
        st.stop()
```

---

## 📈 Analytics & Monitoring

### Add Google Analytics
```python
# In your app
google_analytics = """
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
"""
st.components.v1.html(google_analytics, height=0)
```

### Track Usage
```python
# Log user actions
def log_action(action, details):
    with open('usage.log', 'a') as f:
        f.write(f"{datetime.now()}: {action} - {details}\n")

# Use it
if st.button("Generate Word Cloud"):
    log_action("word_cloud", f"user generated word cloud")
```

---

## 🎓 Educational Deployment (BYU-Idaho)

### For Classroom Use:

1. **Computer Lab Setup:**
```bash
# Install on all machines
pip install -r requirements_streamlit.txt

# Create batch file for students (Windows)
echo "streamlit run streamlit_app.py" > run_app.bat
```

2. **Share via Campus Network:**
- Deploy on campus server
- Students access via internal URL
- No internet required

3. **Student Projects:**
- Each student forks the repo
- Modify for different speakers/topics
- Deploy their own version

### Integration with Course:

```python
# Add course-specific features
if st.sidebar.checkbox("SCM 361 Mode"):
    st.markdown("""
    ### Supply Chain Applications
    - Analyze supplier communications
    - Track terminology evolution
    - Identify trending topics
    """)
```

---

## 🚢 Production Checklist

Before deploying to production:

- [ ] Remove debug mode
- [ ] Add error handling
- [ ] Implement logging
- [ ] Set up backups
- [ ] Add user authentication (if needed)
- [ ] Optimize database queries
- [ ] Test on mobile devices
- [ ] Add usage analytics
- [ ] Create user documentation
- [ ] Set up monitoring alerts

---

## 📞 Support & Resources

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues:** Report bugs in your repo
- **BYU-I IT Support:** For campus deployment

---

## Next Steps

1. **Start Local:** Test the app locally first
2. **Deploy Free:** Use Streamlit Cloud for initial deployment
3. **Gather Feedback:** Share with colleagues/students
4. **Scale Up:** Move to paid hosting if needed
5. **Extend:** Add features based on user feedback

---

*Happy Deploying! 🚀*
