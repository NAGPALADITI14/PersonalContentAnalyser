import os
import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from collections import Counter
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ChromePreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Extended stop words specific to web browsing
        self.custom_stop_words = {'www', 'com', 'http', 'https', 'org', 'html', 'php', 'asp', 
                                 'watch', 'video', 'google', 'search', 'query', 'result'}
        self.stop_words.update(self.custom_stop_words)
    
    def preprocess_chrome_history(self, file_path):
        """Process Chrome history JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            if 'Browser History' not in data:
                print("No browser history found in the JSON file")
                return pd.DataFrame()
            
            history_items = data['Browser History']
            
            # Convert to DataFrame
            df = pd.DataFrame(history_items)
            
            # Convert timestamps
            if 'time_usec' in df.columns:
                # Chrome stores time as microseconds since Jan 1, 1601 UTC
                # Convert to datetime
                df['timestamp'] = pd.to_datetime(df['time_usec'], unit='us',utc=True).dt.tz_localize(None)
                
                # Extract features from timestamps
                df['date'] = df['timestamp'].dt.date
                df['time'] = df['timestamp'].dt.time
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.day_name()
                df['month'] = df['timestamp'].dt.month_name()
                df['year'] = df['timestamp'].dt.year
                
                # Create time bins for time of day analysis
                df['time_of_day'] = pd.cut(
                    df['hour'],
                    bins=[0, 6, 12, 18, 24],
                    labels=['Night (12AM-6AM)', 'Morning (6AM-12PM)', 'Afternoon (12PM-6PM)', 'Evening (6PM-12AM)']
                )
            
            # Extract domain from URL
            if 'url' in df.columns:
                df['domain'] = df['url'].apply(self._extract_domain)
            
            # Clean and process titles
            if 'title' in df.columns:
                df['processed_title'] = df['title'].apply(self._clean_text)
                df['keywords'] = df['processed_title'].apply(self._extract_keywords)
                
                # Categorize the pages based on domain and keywords
                df['category'] = df.apply(self._categorize_chrome_entry, axis=1)
            
            df['platform'] = 'Chrome'
            return df
        except Exception as e:
            print(f"Error processing Chrome history: {e}")
            return pd.DataFrame()
    
    

    def preprocess_chrome_bookmarks(self, file_path):
        """Preprocess Chrome bookmarks HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
            print("Bookmarks HTML parsed successfully:", type(soup))

            # Traverse the <a> tags in the document
            def extract_bookmarks(soup, bookmarks_list):
                for a_tag in soup.find_all('a'):
                    url = a_tag.get('href')
                    if url:
                        title = a_tag.text.strip()
                        bookmark_entry = {
                            'url': url,
                            'domain': self._extract_domain(url),
                            'title': title,
                            'processed_title': self._clean_text(title),
                            'keywords': self._extract_keywords(self._clean_text(title))
                        }
                        bookmarks_list.append(bookmark_entry)

            bookmarks_list = []
            extract_bookmarks(soup, bookmarks_list)

            # Convert to DataFrame
            bookmarks_df = pd.DataFrame(bookmarks_list)

            # Categorize bookmarks
            if 'domain' in bookmarks_df.columns and 'keywords' in bookmarks_df.columns:
                bookmarks_df['category'] = bookmarks_df.apply(self._categorize_chrome_entry, axis=1)

            bookmarks_df['platform'] = 'Chrome Bookmarks'
            return bookmarks_df

        except Exception as e:
            print(f"Error processing Chrome bookmarks: {e}")
            return pd.DataFrame()
    
    def _find_parent_folder(self, dt_tag):
        """Find the parent folder name for a bookmark."""
        # Navigate up to find the closest H3 tag which contains the folder name
        current = dt_tag
        while current and current.name != 'dl':
            current = current.parent
        
        if current and current.previous_sibling:
            h3 = current.previous_sibling.find('h3')
            if h3:
                return h3.text.strip()
        
        return "Root"
    
    def _extract_domain(self, url):
        """Extract domain from URL."""
        try:
            # Simple domain extraction using regex
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                return domain_match.group(1)
        except:
            pass
        return "unknown"
    
    def _clean_text(self, text):
        """Clean and normalize text."""
        if not isinstance(text, str) or not text:
            return ""
        
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_keywords(self, text):
        """Extract meaningful keywords from text."""
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and words with less than 3 characters
        keywords = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return keywords
    
    def _categorize_chrome_entry(self, row):
        """Categorize Chrome entries based on domain and keywords."""
        domain = row.get('domain', '').lower()
        keywords = row.get('keywords', [])
        title = row.get('processed_title', '').lower()
        
        # Define category patterns
        categories = {
            'Social Media': ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'tiktok.com', 'reddit.com', 'quora.com', 'pinterest.com'],
            'Video': ['youtube.com', 'netflix.com', 'hulu.com', 'primevideo.com', 'disneyplus.com', 'twitch.tv', 'vimeo.com'],
            'Shopping': ['amazon.com', 'ebay.com', 'walmart.com', 'etsy.com', 'aliexpress.com', 'shop', 'store', 'buy'],
            'News': ['news', 'cnn.com', 'bbc.com', 'nytimes.com', 'reuters.com', 'foxnews.com', 'washingtonpost.com'],
            'Technology': ['github.com', 'stackoverflow.com', 'medium.com', 'tech', 'code', 'programming', 'developer'],
            'Education': ['edu', 'course', 'learn', 'tutorial', 'university', 'school', 'academy', 'coursera.org', 'udemy.com', 'edx.org', 'Khan Academy'],
            'Entertainment': ['game', 'movie', 'music', 'entertainment', 'spotify.com', 'imdb.com', 'soundcloud.com'],
            'Finance': ['finance', 'bank', 'money', 'invest', 'stock', 'crypto', 'financial', 'paypal.com', 'blockchain'],
            'Travel': ['travel', 'hotel', 'flight', 'booking.com', 'airbnb.com', 'expedia.com', 'tripadvisor.com', 'vacation'],
            'Health': ['health', 'fitness', 'medical', 'doctor', 'workout', 'diet', 'wellness', 'exercise'],
            'Food': ['food', 'recipe', 'cooking', 'restaurant', 'meal', 'diet', 'nutrition'],
            'Email': ['mail', 'gmail.com', 'outlook.com', 'yahoo.com', 'email']
        }
        
        # Check domain first
        for category, patterns in categories.items():
            for pattern in patterns:
                if pattern in domain:
                    return category
        
        # Then check keywords and title
        keyword_text = ' '.join(keywords) + ' ' + title
        for category, patterns in categories.items():
            for pattern in patterns:
                if pattern in keyword_text:
                    return category
        
        return "Other"
    
    def analyze_browsing_habits(self, df):
        """Analyze browsing habits based on Chrome history data."""
        if df.empty:
            return {}
        
        analysis = {}
        
        # Most visited domains
        if 'domain' in df.columns:
            top_domains = df['domain'].value_counts().head(10).to_dict()
            analysis['top_domains'] = top_domains
        
        # Category distribution
        if 'category' in df.columns:
            category_counts = df['category'].value_counts().to_dict()
            analysis['category_distribution'] = category_counts
        
        # Time analysis
        if 'hour' in df.columns:
            hour_counts = df['hour'].value_counts().sort_index().to_dict()
            analysis['hourly_distribution'] = hour_counts
        
        if 'day_of_week' in df.columns:
            day_counts = df['day_of_week'].value_counts().to_dict()
            analysis['daily_distribution'] = day_counts
        
        # Potential focus or distraction metrics
        if 'category' in df.columns:
            productive_categories = ['Education', 'Technology', 'News', 'Finance', 'Health']
            entertainment_categories = ['Social Media', 'Video', 'Entertainment', 'Shopping']
            
            productive_count = sum(df['category'].isin(productive_categories))
            entertainment_count = sum(df['category'].isin(entertainment_categories))
            total_count = len(df)
            
            if total_count > 0:
                analysis['productivity_ratio'] = productive_count / total_count
                analysis['entertainment_ratio'] = entertainment_count / total_count
        
        return analysis
    
    def get_usage_patterns(self, df):
        """Extract usage patterns for visualization."""
        if df.empty:
            return {}
        
        patterns = {}
        
        # Hourly distribution
        if 'hour' in df.columns:
            hour_counts = df['hour'].value_counts().sort_index()
            patterns['hourly'] = {
                'labels': [f"{h}:00" for h in hour_counts.index],
                'values': hour_counts.values.tolist()
            }
        
        # Monthly distribution
        if 'month' in df.columns:
            # Define month order
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            month_counts = df['month'].value_counts()
            ordered_months = []
            ordered_counts = []
            
            for month in month_order:
                if month in month_counts:
                    ordered_months.append(month)
                    ordered_counts.append(month_counts[month])
            
            patterns['monthly'] = {
                'labels': ordered_months,
                'values': ordered_counts
            }
        
        # Daily distribution
        if 'day_of_week' in df.columns:
            # Define day order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            day_counts = df['day_of_week'].value_counts()
            ordered_days = []
            ordered_counts = []
            
            for day in day_order:
                if day in day_counts:
                    ordered_days.append(day)
                    ordered_counts.append(day_counts[day])
            
            patterns['daily'] = {
                'labels': ordered_days,
                'values': ordered_counts
            }
        
        return patterns