import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import streamlit as st
import requests
from time import sleep
from random import uniform
nltk.download('punkt')
nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Extended stop words specific to web browsing
        self.custom_stop_words = {'www', 'com', 'http', 'https', 'org', 'html', 'php', 'asp', 
                                 'watch', 'video', 'youtube', 'google', 'search', 'query', 'result'}
        self.stop_words.update(self.custom_stop_words)
    
    def preprocess_youtube_watch_history(self, file_path, limit=None):
        """Process YouTube watch history HTML file with performance optimizations."""
        import time
        start_time = time.time()

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                print("[INFO] Reading and parsing HTML...")
                soup = BeautifulSoup(file.read(), 'lxml')  # ✅ Faster parser
            print("[INFO] HTML parsed. Extracting entries...")

            # Extract video entries
            entries = []
            items = soup.find_all('div', class_='content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1')
            if limit:
                items = items[:limit]  # ✅ Optional: speed up testing
            print(f"[INFO] Found {len(items)} items to process.")

            for idx, item in enumerate(items):
                # Extract title and URL
                title_elem = item.find('a')
                if not title_elem:
                    continue

                title = title_elem.text.strip()
                url = title_elem.get('href', '')
                video_id = self._extract_youtube_video_id(url)

                # Extract timestamp
                text = item.get_text()
                timestamp_match = re.search(r'(\d{1,2} \w{3} \d{4}, \d{2}:\d{2}:\d{2} GMT[+-]\d{2}:\d{2})', text)
                timestamp = None
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    try:
                        timestamp = pd.to_datetime(timestamp_str, format="%d %b %Y, %H:%M:%S GMT%z")
                    except Exception as e:
                        print(f"[WARN] Failed to parse timestamp: {timestamp_str} | Error: {e}")

                # Extract channel if available
                # channel_elem = item.find(text=re.compile(r'Channel:'))
                # channel = None
                # if channel_elem:
                #     channel_match = re.search(r'Channel: (.*)', channel_elem)
                #     if channel_match:
                #         channel = channel_match.group(1).strip()
                # Extract channel if available
                channel = None
                for span in item.find_all('span'):
                    if 'Channel:' in span.text:
                        channel = span.text.replace('Channel:', '').strip()
                        break

                entries.append({
                    'title': title,
                    'url': url,
                    'video_id': video_id,
                    'timestamp': timestamp,
                    'channel': channel,
                    'platform': 'YouTube'
                })

                if idx % 500 == 0:
                    print(f"[DEBUG] Processed {idx} entries...")

            print("[INFO] Converting to DataFrame...")
            df = pd.DataFrame(entries)

            # Extract datetime features
            if 'timestamp' in df.columns and not df['timestamp'].isnull().all():
                df['date'] = df['timestamp'].dt.date
                df['time'] = df['timestamp'].dt.time
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.day_name()
                df['month'] = df['timestamp'].dt.month_name()
                df['year'] = df['timestamp'].dt.year

                df['time_of_day'] = pd.cut(
                    df['hour'],
                    bins=[0, 6, 12, 18, 24],
                    labels=['Night (12AM-6AM)', 'Morning (6AM-12PM)', 'Afternoon (12PM-6PM)', 'Evening (6PM-12AM)']
                )

            # Keyword extraction (if needed)
            if 'title' in df.columns and not df.empty:
                print("[INFO] Cleaning and extracting keywords...")
                df['processed_title'] = df['title'].apply(self._clean_text)
                df['keywords'] = df['processed_title'].apply(self._extract_keywords)

            duration = time.time() - start_time
            print(f"[SUCCESS] Processing completed in {duration:.2f} seconds.")
            return df

        except Exception as e:
            print(f"[ERROR] Error processing YouTube watch history: {e}")
            return pd.DataFrame()

    def preprocess_youtube_search_history(self, file_path):
        """Process YouTube search history HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
            print("HTML parsed successfully. Checking structure...")

            # Extract search entries
            entries = []

            # ✅ Use CSS selector to robustly match div with all required classes
            for item in soup.select('div.content-cell.mdl-cell.mdl-cell--6-col.mdl-typography--body-1'):
                # Extract search query from the <a> tag
                link = item.find('a')
                if not link:
                    continue
                query = link.get_text().strip()

                # Extract timestamp from next text element after <br>
                br = item.find('br')
                timestamp = None
                if br and br.next_sibling:
                    timestamp_str = br.next_sibling.strip()
                    timestamp_match = re.search(r'\d{1,2} \w{3} \d{4}, \d{2}:\d{2}:\d{2} GMT[+-]\d{2}:\d{2}', timestamp_str)
                    if timestamp_match:
                        try:
                            timestamp = pd.to_datetime(
                                timestamp_match.group(0),
                                format="%d %b %Y, %H:%M:%S GMT%z"
                            )
                        except Exception as e:
                            print(f"Error parsing timestamp: {e}")

                entries.append({
                    'query': query,
                    'timestamp': timestamp,
                    'platform': 'YouTube Search'
                })

            # Convert to DataFrame
            df = pd.DataFrame(entries)
            if df.empty:
                print("No valid search entries found in the file.")
                return pd.DataFrame()

            # Extract features from timestamps
            if 'timestamp' in df.columns and not df['timestamp'].isnull().all():
                df['date'] = df['timestamp'].dt.date
                df['time'] = df['timestamp'].dt.time
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.day_name()
                df['month'] = df['timestamp'].dt.month_name()
                df['year'] = df['timestamp'].dt.year

                # Time bins
                df['time_of_day'] = pd.cut(
                    df['hour'],
                    bins=[0, 6, 12, 18, 24],
                    labels=['Night (12AM-6AM)', 'Morning (6AM-12PM)', 'Afternoon (12PM-6PM)', 'Evening (6PM-12AM)']
                )

            # Keyword extraction
            if 'query' in df.columns and len(df) > 0:
                df['processed_query'] = df['query'].apply(self._clean_text)
                df['keywords'] = df['processed_query'].apply(self._extract_keywords)

            return df

        except Exception as e:
            print(f"Error processing YouTube search history: {e}")
            return pd.DataFrame()

    
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
                df['timestamp'] = pd.to_datetime(df['time_usec'], unit='us')
                
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
        
    def preprocess_chrome_bookmarks(file_path):
        """Preprocess Chrome bookmarks file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                bookmarks_data = json.load(f)
            print("Bookmarks data loaded:", type(bookmarks_data))
            
            # Traverse the bookmarks structure
            def extract_bookmarks(node, bookmarks_list):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key == 'children' and isinstance(value, list):
                            for child in value:
                                extract_bookmarks(child, bookmarks_list)
                        elif key == 'url' and isinstance(value, str):
                            bookmarks_list.append(value)
                elif isinstance(node, list):
                    for item in node:
                        extract_bookmarks(item, bookmarks_list)
                else:
                    print(f"Unexpected node type: {type(node)} - {node}")

            bookmarks_list = []
            extract_bookmarks(bookmarks_data, bookmarks_list)
            
            # Convert to DataFrame
            bookmarks_df = pd.DataFrame({'url': bookmarks_list})
            return bookmarks_df
        except Exception as e:
            print(f"Error processing Chrome bookmarks: {e}")
            return None
    

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
    
    def _extract_youtube_video_id(self, url):
        """Extract YouTube video ID from URL."""
        try:
            # Match YouTube video ID patterns
            patterns = [
                r'youtube\.com/watch\?v=([^&]+)',  # Standard watch URL
                r'youtu\.be/([^?]+)',              # Short URL
                r'youtube\.com/embed/([^?]+)',     # Embed URL
                r'youtube\.com/v/([^?]+)'          # Old style URL
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)
        except:
            pass
        return None
    
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
    
    def analyze_topic_diversity(self, df, text_column):
        """Analyze topic diversity using TF-IDF and clustering."""
        if text_column not in df.columns or df.empty:
            return None, None, None
        
        # Filter out rows with empty text
        text_data = df[df[text_column].notna() & (df[text_column] != '')]
        
        if len(text_data) < 5:  # Need enough data for meaningful clustering
            return None, None, None
        
        # Create TF-IDF matrix
        tfidf_vectorizer = TfidfVectorizer(max_features=500)
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data[text_column])
        
        # Determine optimal number of clusters (max 10)
        max_clusters = min(10, len(text_data) // 5)
        if max_clusters < 2:
            max_clusters = 2
            
        inertia = []
        for k in range(2, max_clusters+1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix)
            inertia.append(kmeans.inertia_)
        
        # Find elbow point (simplified)
        k = 2
        if len(inertia) > 1:
            diffs = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
            if diffs:
                k = np.argmin(diffs) + 3  # +3 because we started from k=2 and need to account for 0-indexing
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Get top terms for each cluster
        features = tfidf_vectorizer.get_feature_names_out()
        cluster_terms = {}
        cluster_distribution = {}
        
        for i in range(k):
            # Get top terms for this cluster
            cluster_indices = [idx for idx, label in enumerate(clusters) if label == i]
            if not cluster_indices:
                continue
                
            cluster_tfidf = tfidf_matrix[cluster_indices].toarray().mean(axis=0)
            top_indices = cluster_tfidf.argsort()[-10:][::-1]
            top_terms = [features[idx] for idx in top_indices]
            
            cluster_terms[f"Topic {i+1}"] = top_terms
            cluster_distribution[f"Topic {i+1}"] = len(cluster_indices) / len(clusters)
        
        # Add cluster info to dataframe
        text_data_with_clusters = text_data.copy()
        text_data_with_clusters['topic_cluster'] = clusters
        
        return cluster_terms, cluster_distribution, text_data_with_clusters
    
    def identify_biases(self, df, column='category'):
        """Identify potential biases in content consumption."""
        if column not in df.columns or df.empty:
            return {}
        
        # Calculate frequency distribution
        value_counts = df[column].value_counts()
        total = len(df)
        
        # Calculate percentages
        percentages = value_counts / total * 100
        
        # Find overrepresented and underrepresented categories
        threshold_high = 25  # Categories with > 25% are considered overrepresented
        threshold_low = 5    # Categories with < 5% are considered underrepresented
        
        biases = {
            'overrepresented': [],
            'underrepresented': [],
            'distribution': {}
        }
        
        for category, percentage in percentages.items():
            biases['distribution'][category] = percentage
            
            if percentage > threshold_high:
                biases['overrepresented'].append(category)
            elif percentage < threshold_low:
                biases['underrepresented'].append(category)
        
        return biases
    
    def generate_recommendations(self, biases, user_interests=None):
        """Generate content diversity recommendations based on biases."""
        recommendations = []
        
        if not biases or 'overrepresented' not in biases or 'underrepresented' not in biases:
            return recommendations
        
        # Base recommendations on overrepresented and underrepresented categories
        if biases['overrepresented']:
            recommendations.append({
                'type': 'Diversify',
                'message': f"You might be overconsuming content in these categories: {', '.join(biases['overrepresented'])}. Try to diversify your content consumption."
            })
        
        if biases['underrepresented'] and len(biases['distribution']) > 5:
            under_repr = [cat for cat in biases['underrepresented'] 
                         if cat in ['Education', 'Technology', 'News', 'Health', 'Finance', 'Science']]
            if under_repr:
                recommendations.append({
                    'type': 'Explore',
                    'message': f"Consider exploring more content in these valuable categories: {', '.join(under_repr)}."
                })
        
        # Add specific recommendations based on user interests and biases
        if user_interests:
            for interest in user_interests:
                opposing_viewpoints = self._get_opposing_viewpoints(interest)
                if opposing_viewpoints:
                    recommendations.append({
                        'type': 'Balance',
                        'message': f"Since you're interested in {interest}, consider exploring these alternative perspectives: {', '.join(opposing_viewpoints)}."
                    })
        
        return recommendations
    
    def _get_opposing_viewpoints(self, interest):
        """Get opposing or complementary viewpoints for a given interest."""
        # Map of interests to complementary or opposing viewpoints
        viewpoint_map = {
            'Technology': ['Ethics in Technology', 'Digital Minimalism', 'Traditional Skills'],
            'Politics': ['Multiple political perspectives', 'Political Philosophy', 'Policy Details vs Political Drama'],
            'Finance': ['Economic History', 'Alternative Economic Models', 'Economics from Different Global Perspectives'],
            'Health': ['Traditional Medicine', 'Mental Health', 'Preventive Health'],
            'Entertainment': ['Educational Content', 'Documentary', 'Classic Literature'],
            'Social Media': ['Digital Detox', 'In-Person Social Interaction', 'Long-Form Content'],
            'News': ['Historical Context', 'International Perspectives', 'Slow Journalism'],
            'Shopping': ['Minimalism', 'Sustainable Consumption', 'DIY and Repair'],
            'Gaming': ['Outdoor Activities', 'Creative Hobbies', 'Game Design Theory']
        }
        
        # Try direct match first
        if interest in viewpoint_map:
            return viewpoint_map[interest]
        
        # Try partial match
        for key in viewpoint_map:
            if key.lower() in interest.lower():
                return viewpoint_map[key]
        
        return []
    
    def generate_nutrition_label(self, df):
        """Generate an 'intellectual nutrition label' based on consumption patterns."""
        if df.empty:
            return {}
        
        nutrition_label = {}
        
        # Calculate time spent in each category if timestamp data is available
        if 'category' in df.columns and 'timestamp' in df.columns:
            # Group by category and calculate statistics
            category_stats = df.groupby('category').agg({
                'timestamp': 'count'
            }).reset_index()
            
            category_stats = category_stats.rename(columns={'timestamp': 'count'})
            category_stats['percentage'] = category_stats['count'] / category_stats['count'].sum() * 100
            
            nutrition_label['category_breakdown'] = category_stats.to_dict('records')
        
        # Add time of day analysis
        if 'time_of_day' in df.columns:
            time_of_day_counts = df['time_of_day'].value_counts()
            nutrition_label['time_of_day'] = {
                tod: count for tod, count in time_of_day_counts.items()
            }
        
        # Add day of week analysis
        if 'day_of_week' in df.columns:
            day_counts = df['day_of_week'].value_counts()
            nutrition_label['day_of_week'] = {
                day: count for day, count in day_counts.items()
            }
        
        # Add month analysis
        if 'month' in df.columns:
            month_counts = df['month'].value_counts()
            nutrition_label['month'] = {
                month: count for month, count in month_counts.items()
            }
        
        # Add topic diversity metrics if we have content data
        if 'keywords' in df.columns and not df['keywords'].isnull().all():
            # Flatten all keywords
            all_keywords = [kw for sublist in df['keywords'].dropna() for kw in sublist]
            
            if all_keywords:
                keyword_counts = Counter(all_keywords)
                top_keywords = dict(keyword_counts.most_common(20))
                nutrition_label['top_keywords'] = top_keywords
                
                # Calculate entropy as a diversity measure
                total_kw = sum(keyword_counts.values())
                probabilities = [count/total_kw for count in keyword_counts.values()]
                entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                
                # Normalize entropy between 0 and 1 (1 being most diverse)
                max_entropy = np.log2(len(keyword_counts))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                nutrition_label['topic_diversity_score'] = normalized_entropy
                nutrition_label['diversity_interpretation'] = self._interpret_diversity_score(normalized_entropy)
            
        return nutrition_label
    
    def _interpret_diversity_score(self, score):
        """Interpret a diversity score with descriptive text."""
        if score >= 0.8:
            return "Excellent diversity - You consume a wide variety of topics"
        elif score >= 0.6:
            return "Good diversity - Your content consumption is fairly varied"
        elif score >= 0.4:
            return "Moderate diversity - Consider expanding your range of topics"
        elif score >= 0.2:
            return "Limited diversity - Your content is concentrated in a few topics"
        else:
            return "Very limited diversity - You focus on a narrow range of topics"
    
    def generate_wordcloud(self, df, column='keywords'):
        """Generate word cloud data from keywords or text."""
        if column not in df.columns or df.empty:
            return None
        
        # For keywords column (list of keywords)
        if column == 'keywords':
            all_keywords = [kw for sublist in df[column].dropna() for kw in sublist if isinstance(sublist, list)]
            text = ' '.join(all_keywords)
        # For text columns
        else:
            text = ' '.join(df[column].dropna().astype(str).tolist())
        
        if not text:
            return None
        
        # Generate wordcloud
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                             max_words=100, contour_width=3, contour_color='steelblue')
        wordcloud.generate(text)
        
        # Convert to image
        img = wordcloud.to_image()
        
        # Save to BytesIO object
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Convert to base64 for embedding in HTML
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
        
        return img_base64
    
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
                'values' : ordered_counts
            }
            return patterns