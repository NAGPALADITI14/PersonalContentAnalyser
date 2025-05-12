import os
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import time
from io import BytesIO
import base64
from wordcloud import WordCloud

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

class YouTubePreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Extended stop words specific to web browsing
        self.custom_stop_words = {'www', 'com', 'http', 'https', 'org', 'html', 'php', 'asp', 
                                 'watch', 'video', 'youtube', 'google', 'search', 'query', 'result'}
        self.stop_words.update(self.custom_stop_words)

        
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
    
    def preprocess_youtube_watch_history(self, file_path, limit=None):
        """Process YouTube watch history HTML file with performance optimizations."""
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
                
                # Categorize videos based on title keywords
                df['category'] = df.apply(self._categorize_youtube_entry, axis=1)

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
    
    def _categorize_youtube_entry(self, row):
        """Categorize YouTube videos based on title keywords and channel."""
        title = row.get('processed_title', '').lower()
        keywords = row.get('keywords', [])
        channel = str(row.get('channel', '')).lower()
        
        # Define category patterns for YouTube
        categories = {
            'Music': ['music', 'song', 'concert', 'remix', 'album', 'band', 'lyric', 'audio', 'tune', 'melody', 'instrumental'],
            'Gaming': ['game', 'gaming', 'playthrough', 'walkthrough', 'minecraft', 'fortnite', 'gameplay', 'stream', 'streamer'],
            'Tech': ['tech', 'technology', 'review', 'unboxing', 'gadget', 'smartphone', 'computer', 'programming', 'coding', 'developer', 'software'],
            'Education': ['learn', 'tutorial', 'education', 'course', 'lecture', 'how to', 'explained', 'university', 'school', 'academic'],
            'Entertainment': ['funny', 'comedy', 'prank', 'challenge', 'reaction', 'vlog', 'entertainment', 'laugh', 'parody', 'skit'],
            'News': ['news', 'politics', 'report', 'update', 'current events', 'breaking', 'interview', 'analysis', 'discussion'],
            'Science': ['science', 'experiment', 'physics', 'chemistry', 'biology', 'space', 'astronomy', 'research', 'discovery'],
            'Cooking': ['recipe', 'cooking', 'baking', 'food', 'cook', 'chef', 'dish', 'meal', 'cuisine', 'kitchen'],
            'Fitness': ['fitness', 'workout', 'exercise', 'gym', 'training', 'health', 'sport', 'yoga', 'cardio', 'strength'],
            'DIY': ['diy', 'craft', 'make', 'build', 'how to make', 'project', 'handmade', 'homemade', 'creation'],
            'Fashion': ['fashion', 'style', 'makeup', 'beauty', 'hair', 'clothing', 'outfit', 'dress', 'accessories'],
            'Travel': ['travel', 'tour', 'vacation', 'trip', 'journey', 'destination', 'explore', 'adventure', 'vlog'],
            'Automotive': ['car', 'auto', 'vehicle', 'drive', 'driving', 'review', 'racing', 'engine', 'motorcycle']
        }
        
        # Check combined text for category matches
        combined_text = title + ' ' + ' '.join(keywords) + ' ' + channel
        
        for category, patterns in categories.items():
            for pattern in patterns:
                if pattern in combined_text:
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
    
    def analyze_youtube_habits(self, watch_df, search_df=None):
        """Analyze YouTube watch and search habits."""
        analysis = {}
        
        # Analyze watch history
        if not watch_df.empty:
            # Top channels
            if 'channel' in watch_df.columns and watch_df['channel'].notna().any():
                top_channels = watch_df['channel'].value_counts().head(10).to_dict()
                analysis['top_channels'] = top_channels
            
            # Time analysis
            if 'hour' in watch_df.columns:
                hour_counts = watch_df['hour'].value_counts().sort_index().to_dict()
                analysis['watch_hour_distribution'] = hour_counts
            
            if 'day_of_week' in watch_df.columns:
                day_counts = watch_df['day_of_week'].value_counts().to_dict()
                analysis['watch_day_distribution'] = day_counts
            
            # Category distribution
            if 'category' in watch_df.columns:
                category_counts = watch_df['category'].value_counts().to_dict()
                analysis['category_distribution'] = category_counts
            
            # Extract trending topics
            if 'keywords' in watch_df.columns and watch_df['keywords'].notna().any():
                all_keywords = [kw for sublist in watch_df['keywords'] if isinstance(sublist, list) for kw in sublist]
                keyword_counts = {}
                for kw in all_keywords:
                    if kw in keyword_counts:
                        keyword_counts[kw] += 1
                    else:
                        keyword_counts[kw] = 1
                
                # Sort by frequency and get top 20
                sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
                analysis['trending_topics'] = dict(sorted_keywords)
            
            # Video duration analysis
            if 'duration_seconds' in watch_df.columns:
                # Calculate average watch time
                analysis['avg_video_duration'] = watch_df['duration_seconds'].mean()
                
                # Categorize videos by duration
                duration_bins = [0, 60, 300, 600, 1200, 1800, 3600, float('inf')]
                duration_labels = ['<1min', '1-5min', '5-10min', '10-20min', '20-30min', '30-60min', '>60min']
                watch_df['duration_category'] = pd.cut(watch_df['duration_seconds'], bins=duration_bins, labels=duration_labels)
                duration_distribution = watch_df['duration_category'].value_counts().to_dict()
                analysis['duration_distribution'] = duration_distribution
            
            # Watch patterns over time
            if 'date' in watch_df.columns:
                # Group by date and count videos watched per day
                daily_counts = watch_df.groupby('date').size().to_dict()
                analysis['daily_watch_counts'] = daily_counts
                
                # Calculate average videos per day
                analysis['avg_videos_per_day'] = watch_df['date'].value_counts().mean()
                
                # Find most active days (top 5)
                most_active_days = watch_df['date'].value_counts().head(5).to_dict()
                analysis['most_active_days'] = most_active_days
            
            # Title analysis for common themes
            if 'title' in watch_df.columns and watch_df['title'].notna().any():
                # Simple word frequency analysis
                import re
                from collections import Counter
                
                # Common stop words to filter out
                stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'from'}
                
                # Extract words from titles
                all_words = []
                for title in watch_df['title']:
                    if isinstance(title, str):
                        # Convert to lowercase and remove punctuation
                        words = re.findall(r'\w+', title.lower())
                        # Filter out stop words and short words
                        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
                        all_words.extend(filtered_words)
                
                # Get most common words
                word_counts = Counter(all_words).most_common(20)
                analysis['common_title_themes'] = dict(word_counts)
        
        # Analyze search history if available
        if search_df is not None and not search_df.empty:
            # Top search terms
            if 'search_term' in search_df.columns:
                top_searches = search_df['search_term'].value_counts().head(20).to_dict()
                analysis['top_searches'] = top_searches
            
            # Search time patterns
            if 'hour' in search_df.columns:
                search_hour_counts = search_df['hour'].value_counts().sort_index().to_dict()
                analysis['search_hour_distribution'] = search_hour_counts
            
            if 'day_of_week' in search_df.columns:
                search_day_counts = search_df['day_of_week'].value_counts().to_dict()
                analysis['search_day_distribution'] = search_day_counts
            
            # Search patterns over time
            if 'date' in search_df.columns:
                daily_search_counts = search_df.groupby('date').size().to_dict()
                analysis['daily_search_counts'] = daily_search_counts
            
            # Correlation between searches and watches
            if 'date' in watch_df.columns and 'date' in search_df.columns:
                try:
                    # Group by date and count for both dataframes
                    watch_counts = watch_df.groupby('date').size().to_dict()
                    search_counts = search_df.groupby('date').size().to_dict()
                    
                    # Find common dates
                    common_dates = set(watch_counts.keys()).intersection(set(search_counts.keys()))
                    
                    # Create lists of counts for common dates
                    watch_list = [watch_counts.get(date, 0) for date in common_dates]
                    search_list = [search_counts.get(date, 0) for date in common_dates]
                    
                    # Calculate correlation if there are enough data points
                    if len(watch_list) > 5:
                        import numpy as np
                        correlation = np.corrcoef(watch_list, search_list)[0, 1]
                        analysis['search_watch_correlation'] = correlation
                except Exception as e:
                    analysis['search_watch_correlation_error'] = str(e)
        
        # Generate insights based on the analysis
        insights = []
        
        # Channel loyalty insights
        if 'top_channels' in analysis and analysis['top_channels']:
            total_videos = len(watch_df)
            top_channel, top_count = list(analysis['top_channels'].items())[0]
            top_channel_percentage = (top_count / total_videos) * 100
            
            if top_channel_percentage > 30:
                insights.append(f"Strong channel loyalty: {top_channel_percentage:.1f}% of your watched videos are from {top_channel}")
            
            if len(analysis['top_channels']) <= 5 and sum(analysis['top_channels'].values()) / total_videos > 0.7:
                insights.append("You have a narrow channel focus, watching primarily a small set of channels")
        
        # Time pattern insights
        if 'watch_hour_distribution' in analysis:
            # Find peak watching hours
            peak_hours = sorted(analysis['watch_hour_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]
            peak_hours_str = ", ".join([f"{hour}:00" for hour, _ in peak_hours])
            insights.append(f"Your peak YouTube watching times are around: {peak_hours_str}")
        
        # Category preferences
        if 'category_distribution' in analysis and analysis['category_distribution']:
            top_category, category_count = list(analysis['category_distribution'].items())[0]
            insights.append(f"Your most watched content category is {top_category}")
        
        # Content duration preferences
        if 'duration_distribution' in analysis:
            # Find preferred duration
            top_duration = max(analysis['duration_distribution'].items(), key=lambda x: x[1])
            insights.append(f"You tend to prefer videos that are {top_duration[0]}")
        
        analysis['insights'] = insights
        
        return analysis