import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class IntellectualNutrition:
    """Class for analyzing intellectual nutrition based on content consumption data."""
    
    def __init__(self):
        """Initialize the IntellectualNutrition analyzer."""
        self.bias_keywords = {
            'political': {
                'liberal': ['liberal', 'democrat', 'progressive', 'left-wing', 'socialism', 
                           'leftist', 'social justice', 'blm', 'gun control', 'democratic'],
                'conservative': ['conservative', 'republican', 'right-wing', 'trump', 'patriot', 
                               'traditional values', 'capitalism', 'libertarian', 'evangelical']
            },
            'sentiment': {
                'positive': ['solution', 'success', 'improvement', 'growth', 'opportunity', 
                           'innovative', 'breakthrough', 'advancement', 'benefit', 'progress'],
                'negative': ['problem', 'crisis', 'failure', 'decline', 'threat', 
                           'danger', 'disaster', 'risk', 'downfall', 'controversy']
            },
            'perspective': {
                'mainstream': ['official', 'mainstream', 'expert', 'verified', 'authority', 
                             'consensus', 'established', 'conventional', 'recognized'],
                'alternative': ['alternative', 'independent', 'conspiracy', 'underground', 
                              'whistleblower', 'unconventional', 'fringe', 'hidden', 'censored']
            }
        }
        
        # Define content categories with educational value ratings
        self.educational_value = {
            'Education': 0.9,
            'Science & Technology': 0.85,
            'News & Politics': 0.7,
            'Documentary': 0.8,
            'History': 0.85,
            'Programming': 0.85,
            'Health': 0.75,
            'Business': 0.75,
            'Arts & Culture': 0.7,
            'Personal Development': 0.65,
            'Travel': 0.6,
            'Sports': 0.5,
            'Music': 0.4,
            'Entertainment': 0.3,
            'Gaming': 0.4,
            'Comedy': 0.3,
            'Social Media': 0.2,
            'Shopping': 0.1
        }
        
        # Define optimal content balance percentages
        self.optimal_balance = {
            'Educational': 0.40,  # Educational content (education, science, documentation)
            'Informational': 0.30,  # Information gathering (news, tech updates, trends)
            'Entertainment': 0.20,  # Entertainment (music, movies, comedy)
            'Social': 0.10,  # Social interaction & networking
        }
        
        # Content type mapping
        self.content_type_mapping = {
            'Educational': ['Education', 'Science & Technology', 'Documentary', 'History', 
                           'Programming', 'Arts & Culture'],
            'Informational': ['News & Politics', 'Health', 'Business', 'Travel', 'Personal Development'],
            'Entertainment': ['Music', 'Entertainment', 'Gaming', 'Comedy', 'Sports'],
            'Social': ['Social Media', 'Shopping']
        }
    
    def analyze_intellectual_nutrition(self, data):
        """Analyze intellectual nutrition from provided data."""
        results = {}
        
        # Combine all relevant data
        combined_data = self._combine_content_data(data)
        
        if combined_data is None or combined_data.empty:
            return None
        
        # Calculate content consumption balance
        results['content_balance'] = self._analyze_content_balance(combined_data)
        
        # Calculate bias metrics
        results['bias_metrics'] = self._analyze_bias(combined_data)
        
        # Calculate topic diversity
        results['topic_diversity'] = self._analyze_topic_diversity(combined_data)
        
        # Calculate usage patterns
        results['usage_patterns'] = self._analyze_usage_patterns(combined_data)
        
        return results
    
    def _combine_content_data(self, data):
        """Combine relevant content data from different sources."""
        content_data = []
        
        for source, df in data.items():
            if df is None or df.empty:
                continue
                
            required_columns = ['title', 'timestamp']
            
            # Skip if required columns are not present
            if not all(col in df.columns for col in required_columns):
                continue
            
            # Select and standardize relevant columns
            source_data = df[required_columns].copy()
            source_data['source'] = source
            
            # Add category if available
            if 'category' in df.columns:
                source_data['category'] = df['category']
            else:
                source_data['category'] = 'Unknown'
                
            # Add description or content if available
            if 'description' in df.columns:
                source_data['text_content'] = df['description']
            elif 'content' in df.columns:
                source_data['text_content'] = df['content']
            else:
                source_data['text_content'] = df['title']
                
            # Add domain for web content
            if 'domain' in df.columns:
                source_data['domain'] = df['domain']
            else:
                source_data['domain'] = None
                
            content_data.append(source_data)
        
        if not content_data:
            return None
            
        combined_df = pd.concat(content_data)
        
        # Ensure timestamp is datetime
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
        
        # Remove NaT values if any after coercion
        combined_df = combined_df.dropna(subset=['timestamp'])
        
        # Remove timezone info if present
        combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize(None)
        
        # Add time-based features
        combined_df['date'] = combined_df['timestamp'].dt.date
        combined_df['hour'] = combined_df['timestamp'].dt.hour
        combined_df['day_of_week'] = combined_df['timestamp'].dt.day_name()
        
        # Categorize time of day
        combined_df['time_of_day'] = combined_df['hour'].apply(self._categorize_time_of_day)
        
        return combined_df
    
    def _categorize_time_of_day(self, hour):
        """Categorize hour into time of day."""
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'
    
    def _analyze_content_balance(self, combined_data):
        """Analyze content consumption balance."""
        # Map categories to content types
        def map_to_content_type(category):
            for content_type, categories in self.content_type_mapping.items():
                if category in categories:
                    return content_type
            return 'Entertainment'  # Default to entertainment if unknown
        
        # Map educational value
        def get_educational_value(category):
            return self.educational_value.get(category, 0.3)  # Default to 0.3 if unknown
        
        # Add content type and educational value
        combined_data['content_type'] = combined_data['category'].apply(map_to_content_type)
        combined_data['educational_value'] = combined_data['category'].apply(get_educational_value)
        
        # Calculate content type distribution
        content_type_counts = combined_data['content_type'].value_counts(normalize=True).to_dict()
        
        # Calculate content balance score (how close to optimal)
        balance_score = 0
        content_balance = {}
        
        for content_type, optimal in self.optimal_balance.items():
            actual = content_type_counts.get(content_type, 0)
            
            # Calculate how close to optimal (0 = furthest, 1 = closest)
            difference = abs(optimal - actual)
            type_score = max(0, 1 - difference / optimal)
            
            balance_score += type_score * optimal  # Weight by importance
            content_balance[content_type] = {
                'actual': actual,
                'optimal': optimal,
                'score': type_score
            }
        
        # Calculate overall educational value
        avg_educational_value = combined_data['educational_value'].mean()
        
        # Calculate recency-weighted educational value (more weight to recent content)
        combined_data = combined_data.sort_values('timestamp')
        
        # Calculate time decay weights (more recent = higher weight)
        max_time = combined_data['timestamp'].max()
        time_diff = (max_time - combined_data['timestamp']).dt.total_seconds()
        max_diff = time_diff.max()
        combined_data['recency_weight'] = 1 - (time_diff / max_diff if max_diff > 0 else 0)
        
        # Calculate weighted educational value
        weighted_edu_value = np.average(
            combined_data['educational_value'], 
            weights=combined_data['recency_weight']
        )
        
        # Return results
        return {
            'content_type_distribution': content_type_counts,
            'content_balance': content_balance,
            'balance_score': balance_score,
            'educational_value': avg_educational_value,
            'weighted_educational_value': weighted_edu_value
        }
    
    def _analyze_bias(self, combined_data):
        """Analyze content for potential bias indicators."""
        # Concatenate all text content
        all_text = ' '.join(combined_data['text_content'].astype(str).tolist())
        all_text = all_text.lower()
        
        # Calculate bias metrics
        bias_results = {}
        
        for bias_type, bias_categories in self.bias_keywords.items():
            bias_counts = {}
            bias_percentages = {}
            total_bias_keywords = 0
            
            for category, keywords in bias_categories.items():
                count = sum(all_text.count(keyword) for keyword in keywords)
                bias_counts[category] = count
                total_bias_keywords += count
            
            # Calculate percentages
            if total_bias_keywords > 0:
                for category, count in bias_counts.items():
                    bias_percentages[category] = count / total_bias_keywords
            else:
                for category in bias_categories:
                    bias_percentages[category] = 0
            
            # Calculate bias score (0 = balanced, 1 = extremely biased)
            if len(bias_categories) <= 1:
                bias_score = 0
            else:
                values = list(bias_percentages.values())
                max_value = max(values)
                # Calculate variance from perfect balance
                perfect_balance = 1.0 / len(values)
                variances = [abs(v - perfect_balance) for v in values]
                bias_score = sum(variances) / len(variances) / perfect_balance
            
            bias_results[bias_type] = {
                'counts': bias_counts,
                'percentages': bias_percentages,
                'score': bias_score
            }
        
        # Calculate domain diversity
        domain_diversity = 0
        if 'domain' in combined_data.columns:
            domain_counts = combined_data['domain'].dropna().value_counts()
            
            if len(domain_counts) > 0:
                # Calculate domain diversity using Shannon entropy
                total = sum(domain_counts)
                probabilities = [count / total for count in domain_counts]
                domain_diversity = -sum(p * np.log(p) for p in probabilities if p > 0)
                
                # Normalize to 0-1 range (0 = single source, 1 = perfectly diverse)
                max_entropy = np.log(len(domain_counts))
                domain_diversity = domain_diversity / max_entropy if max_entropy > 0 else 0
        
        # Add overall bias score
        overall_bias_score = sum(b['score'] for b in bias_results.values()) / len(bias_results)
        
        return {
            'bias_metrics': bias_results,
            'overall_bias_score': overall_bias_score,
            'domain_diversity': domain_diversity
        }
    
    def _analyze_topic_diversity(self, combined_data):
        """Analyze topic diversity in content consumption."""
        # Extract and process text for analysis
        texts = combined_data['text_content'].astype(str).tolist()
        titles = combined_data['title'].astype(str).tolist()
        
        # Combine titles and text for better feature extraction
        combined_texts = [f"{titles[i]} {texts[i]}" for i in range(len(texts))]
        
        # Preprocess text
        preprocessed_texts = [self._preprocess_text(text) for text in combined_texts]
        
        # Calculate category diversity
        category_diversity = 0
        if 'category' in combined_data.columns:
            category_counts = combined_data['category'].value_counts()
            total = sum(category_counts)
            probabilities = [count / total for count in category_counts]
            category_diversity = -sum(p * np.log(p) for p in probabilities if p > 0)
            
            # Normalize to 0-1 range
            max_entropy = np.log(len(category_counts))
            category_diversity = category_diversity / max_entropy if max_entropy > 0 else 0
        
        # Calculate semantic diversity using TF-IDF
        semantic_diversity = 0
        if len(preprocessed_texts) >= 2:  # Need at least 2 documents for similarity
            try:
                # Create TF-IDF matrix
                vectorizer = TfidfVectorizer(
                    max_features=1000, 
                    stop_words='english',
                    min_df=2,
                    max_df=0.9
                )
                
                tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
                
                # Calculate pairwise cosine similarity
                cosine_sim = cosine_similarity(tfidf_matrix)
                
                # Calculate average non-diagonal similarity
                n = cosine_sim.shape[0]
                similarity_sum = 0
                similarity_count = 0
                
                for i in range(n):
                    for j in range(i+1, n):
                        similarity_sum += cosine_sim[i, j]
                        similarity_count += 1
                
                avg_similarity = similarity_sum / similarity_count if similarity_count > 0 else 0
                
                # Convert to diversity score (higher similarity = lower diversity)
                semantic_diversity = 1 - avg_similarity
            except:
                # Fallback if TF-IDF fails
                semantic_diversity = 0.5
        
        # Calculate topic diversity score (combine category and semantic diversity)
        topic_diversity_score = (category_diversity + semantic_diversity) / 2
        
        # Extract keywords for topic characterization
        keywords = self._extract_keywords(combined_texts, n=20)
        
        return {
            'category_diversity': category_diversity,
            'semantic_diversity': semantic_diversity,
            'topic_diversity_score': topic_diversity_score,
            'keywords': keywords
        }
    
    def _preprocess_text(self, text):
        """Preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_keywords(self, texts, n=20):
        """Extract top keywords from texts."""
        # Combine texts
        all_text = ' '.join(texts)
        
        # Tokenize
        tokens = word_tokenize(all_text.lower())
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words 
                  and len(token) > 3 and token.isalpha()]
        
        # Count frequencies
        word_counts = Counter(tokens)
        
        # Get top keywords
        return word_counts.most_common(n)
    
    def _analyze_usage_patterns(self, combined_data):
        """Analyze usage patterns over time."""
        usage_patterns = {}
        
        # Time of day distribution
        time_of_day_counts = combined_data['time_of_day'].value_counts().to_dict()
        
        # Day of week distribution
        day_of_week_counts = combined_data['day_of_week'].value_counts().to_dict()
        
        # Calculate peak usage times
        hour_counts = combined_data['hour'].value_counts()
        peak_hour = hour_counts.idxmax() if not hour_counts.empty else None
        
        # Calculate content type by time of day
        content_by_time = {}
        if 'content_type' in combined_data.columns:
            for time_period in ['Morning', 'Afternoon', 'Evening', 'Night']:
                period_data = combined_data[combined_data['time_of_day'] == time_period]
                if not period_data.empty:
                    type_counts = period_data['content_type'].value_counts(normalize=True).to_dict()
                    content_by_time[time_period] = type_counts
        
        # Calculate weekly pattern score (consistency)
        weekly_consistency = 0
        if not combined_data.empty:
            # Group by day of week
            daily_counts = combined_data['day_of_week'].value_counts()
            
            # Calculate coefficient of variation
            mean_count = daily_counts.mean()
            std_count = daily_counts.std()
            
            # Lower variation = higher consistency
            cv = std_count / mean_count if mean_count > 0 else 0
            weekly_consistency = max(0, 1 - cv)
        
        # Calculate content diversity over time
        time_diversity = {}
        try:
            # Group by week and calculate diversity within each week
            combined_data['week'] = combined_data['timestamp'].dt.isocalendar().week
            weekly_groups = combined_data.groupby('week')
            
            for week, group in weekly_groups:
                if 'category' in group.columns:
                    category_counts = group['category'].value_counts()
                    total = sum(category_counts)
                    probabilities = [count / total for count in category_counts]
                    diversity = -sum(p * np.log(p) for p in probabilities if p > 0)
                    max_entropy = np.log(len(category_counts))
                    normalized_diversity = diversity / max_entropy if max_entropy > 0 else 0
                    time_diversity[str(week)] = normalized_diversity
        except:
            # Fallback if time diversity calculation fails
            time_diversity = {}
        
        # Return usage pattern results
        return {
            'time_of_day_distribution': time_of_day_counts,
            'day_of_week_distribution': day_of_week_counts,
            'peak_hour': peak_hour,
            'content_by_time': content_by_time,
            'weekly_consistency': weekly_consistency,
            'time_diversity': time_diversity
        }
    
    def _display_usage_patterns(self, usage_results):
        """Display usage patterns analysis."""
        st.subheader("Usage Patterns Analysis")

        # Time of Day Distribution
        st.markdown("### Time of Day Distribution")
        time_of_day_counts = usage_results.get('time_of_day_distribution', {})
        if time_of_day_counts:
            fig = px.bar(
                x=list(time_of_day_counts.keys()),
                y=list(time_of_day_counts.values()),
                labels={'x': 'Time of Day', 'y': 'Count'},
                title="Content Consumption by Time of Day",
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for time of day distribution.")

        # Day of Week Distribution
        st.markdown("### Day of Week Distribution")
        day_of_week_counts = usage_results.get('day_of_week_distribution', {})
        if day_of_week_counts:
            fig = px.bar(
                x=list(day_of_week_counts.keys()),
                y=list(day_of_week_counts.values()),
                labels={'x': 'Day of Week', 'y': 'Count'},
                title="Content Consumption by Day of Week",
                color_discrete_sequence=['#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for day of week distribution.")

        # Peak Hour
        st.markdown("### Peak Usage Hour")
        peak_hour = usage_results.get('peak_hour', None)
        if peak_hour is not None:
            st.metric("Peak Hour", f"{peak_hour}:00")
        else:
            st.info("No data available for peak usage hour.")

        # Content Type by Time of Day
        st.markdown("### Content Type by Time of Day")
        content_by_time = usage_results.get('content_by_time', {})
        if content_by_time:
            for time_period, type_counts in content_by_time.items():
                st.markdown(f"#### {time_period}")
                fig = px.pie(
                    names=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    title=f"Content Type Distribution - {time_period}",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for content type by time of day.")

        # Weekly Consistency
        st.markdown("### Weekly Consistency")
        weekly_consistency = usage_results.get('weekly_consistency', 0)
        st.metric("Weekly Consistency Score", f"{weekly_consistency:.2f}")
        if weekly_consistency > 0.8:
            st.success("Your content consumption is highly consistent throughout the week.")
        elif weekly_consistency > 0.5:
            st.info("Your content consumption shows moderate consistency.")
        else:
            st.warning("Your content consumption is inconsistent across the week.")

        # Time Diversity
        st.markdown("### Time Diversity")
        time_diversity = usage_results.get('time_diversity', {})
        if time_diversity:
            fig = px.line(
                x=list(time_diversity.keys()),
                y=list(time_diversity.values()),
                labels={'x': 'Week', 'y': 'Diversity Score'},
                title="Content Diversity Over Time",
                color_discrete_sequence=['#00CC96']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for time diversity.")

    def _display_insights_and_recommendations(self, results):
        """Display key insights and actionable recommendations."""
        st.subheader("Key Insights & Recommendations")

        # Content Balance Insights
        st.markdown("### Content Balance Insights")
        balance_score = results['content_balance']['balance_score']
        if balance_score > 0.8:
            st.success("Your content consumption is well-balanced across different types.")
        elif balance_score > 0.5:
            st.info("Your content consumption is moderately balanced, but there is room for improvement.")
        else:
            st.warning("Your content consumption is imbalanced. Consider diversifying your content diet.")

        # Educational Value Insights
        educational_value = results['content_balance']['educational_value']
        weighted_educational_value = results['content_balance']['weighted_educational_value']
        st.markdown(f"**Average Educational Value**: {educational_value:.2f}")
        st.markdown(f"**Recency-Weighted Educational Value**: {weighted_educational_value:.2f}")
        if weighted_educational_value < 0.5:
            st.warning("Consider consuming more educational or informational content to improve intellectual nutrition.")

        # Bias Insights
        st.markdown("### Bias Insights")
        overall_bias_score = results['bias_metrics']['overall_bias_score']
        if overall_bias_score < 0.3:
            st.success("Your content consumption shows minimal bias. Great job maintaining a balanced perspective!")
        elif overall_bias_score < 0.6:
            st.info("Your content consumption shows moderate bias. Consider exploring diverse perspectives.")
        else:
            st.warning("Your content consumption is highly biased. Try to include content from different viewpoints.")

        domain_diversity = results['bias_metrics']['domain_diversity']
        st.markdown(f"**Source Diversity Score**: {domain_diversity:.2f}")
        if domain_diversity < 0.3:
            st.warning("Your content sources are limited. Consider exploring content from a wider range of sources.")
        elif domain_diversity < 0.6:
            st.info("Your content sources are moderately diverse. Aim for greater diversity to reduce echo chamber effects.")
        else:
            st.success("Your content sources are highly diverse. Keep it up!")

        # Topic Diversity Insights
        st.markdown("### Topic Diversity Insights")
        topic_diversity_score = results['topic_diversity']['topic_diversity_score']
        if topic_diversity_score > 0.7:
            st.success("Your content consumption covers a wide range of topics. Excellent diversity!")
        elif topic_diversity_score > 0.4:
            st.info("Your content consumption shows moderate topic diversity. Consider exploring new topics.")
        else:
            st.warning("Your content consumption lacks topic diversity. Try to include content from different areas of interest.")

        # Usage Patterns Insights
        st.markdown("### Usage Patterns Insights")
        weekly_consistency = results['usage_patterns']['weekly_consistency']
        if weekly_consistency > 0.8:
            st.success("Your content consumption is highly consistent throughout the week.")
        elif weekly_consistency > 0.5:
            st.info("Your content consumption shows moderate consistency. Aim for a more regular schedule.")
        else:
            st.warning("Your content consumption is inconsistent. Try to establish a more regular pattern.")

        # Recommendations Section
        st.markdown("### Recommendations")
        recommendations = []

        # Content Balance Recommendations
        if balance_score < 0.8:
            recommendations.append("Diversify your content consumption to include more educational and informational content.")

        # Educational Value Recommendations
        if weighted_educational_value < 0.5:
            recommendations.append("Focus on consuming content with higher educational value, such as documentaries or science-related topics.")

        # Bias Recommendations
        if overall_bias_score > 0.3:
            recommendations.append("Actively seek content from alternative perspectives to reduce bias.")

        # Source Diversity Recommendations
        if domain_diversity < 0.6:
            recommendations.append("Explore content from a wider range of sources to improve source diversity.")

        # Topic Diversity Recommendations
        if topic_diversity_score < 0.7:
            recommendations.append("Expand your content consumption to include topics you haven't explored before.")

        # Usage Patterns Recommendations
        if weekly_consistency < 0.8:
            recommendations.append("Establish a more consistent content consumption schedule to improve regularity.")

        # Display Recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        else:
            st.success("No major improvements needed. Keep up the great work!")


    def show_intellectual_nutrition(self, data):
        """Display intellectual nutrition analysis in Streamlit."""
        st.title("🧠 Intellectual Nutrition Analysis")
        
        if not data:
            st.warning("Please upload data files to begin analysis.")
            st.markdown("""
            ## About Intellectual Nutrition
            
            This analysis helps you understand:
            
            1. **Content Consumption Balance** - How balanced is your digital diet?
            2. **Bias Identification** - Are you exposed to diverse perspectives?
            3. **Topic Diversity Score** - How varied are the topics you consume?
            4. **Usage Pattern Visualization** - When and how you consume content
            
            Upload your data files from the sidebar to get started.
            """)
            return
        
        # Run analysis
        results = self.analyze_intellectual_nutrition(data)
        
        if results is None:
            st.error("Could not analyze intellectual nutrition. Please ensure your data contains necessary information.")
            return
        
        # Show overall nutrition score
        overall_score = self._calculate_overall_score(results)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Overall Intellectual Nutrition Score")
            self._display_nutrition_gauge(overall_score)
        
        with col2:
            st.subheader("Score Components")
            component_scores = {
                "Content Balance": results['content_balance']['balance_score'],
                "Educational Value": results['content_balance']['educational_value'],
                "Perspective Diversity": 1 - results['bias_metrics']['overall_bias_score'],
                "Topic Diversity": results['topic_diversity']['topic_diversity_score']
            }
            
            for name, score in component_scores.items():
                st.metric(name, f"{score:.2f}", help=f"Score from 0 to 1")
        
        # Display tabs for detailed analysis
        tab1, tab2, tab3, tab4 = st.tabs([
            "Content Balance", 
            "Bias Analysis", 
            "Topic Diversity", 
            "Usage Patterns"
        ])
        
        with tab1:
            self._display_content_balance(results['content_balance'])
        
        with tab2:
            self._display_bias_analysis(results['bias_metrics'])
        
        with tab3:
            self._display_topic_diversity(results['topic_diversity'])
        
        with tab4:
            self._display_usage_patterns(results['usage_patterns'])
        
        # Show key insights and recommendations
        st.header("Key Insights & Recommendations")
        self._display_insights_and_recommendations(results)
    
    def _calculate_overall_score(self, results):
        """Calculate overall intellectual nutrition score."""
        # Component weights
        weights = {
            'balance': 0.3,
            'educational': 0.25,
            'diversity': 0.25,
            'bias': 0.2
        }
        
        # Calculate component scores
        balance_score = results['content_balance']['balance_score']
        educational_score = results['content_balance']['weighted_educational_value']
        diversity_score = results['topic_diversity']['topic_diversity_score']
        bias_score = 1 - results['bias_metrics']['overall_bias_score']  # Invert (lower bias = better)
        
        # Calculate weighted overall score
        overall_score = (
            weights['balance'] * balance_score +
            weights['educational'] * educational_score +
            weights['diversity'] * diversity_score +
            weights['bias'] * bias_score
        )
        
        return min(max(overall_score, 0), 1)  # Ensure score is between 0 and 1
    
    def _display_nutrition_gauge(self, score):
        """Display a gauge for the nutrition score."""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Intellectual Nutrition Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 60], 'color': "orange"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': score * 100
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add score interpretation
        if score < 0.3:
            st.error("Your content diet needs significant improvement.")
        elif score < 0.6:
            st.warning("There's room for improvement in your content consumption.")
        elif score < 0.8:
            st.info("You have a reasonably balanced content diet.")
        else:
            st.success("Excellent! You have a well-balanced intellectual diet.")
    
    def _display_content_balance(self, content_results):
        """Display content balance analysis."""
        st.subheader("Content Consumption Balance")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create bar chart comparing actual vs optimal
            actual_values = []
            optimal_values = []
            categories = []
            
            for category, data in content_results['content_balance'].items():
                categories.append(category)
                actual_values.append(data['actual'])
                optimal_values.append(data['optimal'])
            
            fig = go.Figure(data=[
                go.Bar(name='Actual', x=categories, y=actual_values, marker_color='royalblue'),
                go.Bar(name='Optimal', x=categories, y=optimal_values, marker_color='lightgreen')
            ])
            
            fig.update_layout(
                title='Content Type Distribution: Actual vs. Optimal',
                xaxis_title='Content Type',
                yaxis_title='Proportion',
                legend_title='',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Educational Value")
            st.markdown(f"**Average**: {content_results['educational_value']:.2f}")
            st.markdown(f"**Recency-Weighted**: {content_results['weighted_educational_value']:.2f}")
            
            st.markdown("### Content Balance Analysis")
            for category, data in content_results['content_balance'].items():
                status = "✅" if abs(data['actual'] - data['optimal']) / data['optimal'] < 0.3 else "⚠️"
                st.markdown(f"{status} **{category}**: {data['actual']:.0%} (Target: {data['optimal']:.0%})")
            
            st.markdown(f"**Overall Balance Score**: {content_results['balance_score']:.2f}")
    
    def _display_bias_analysis(self, bias_results):
        """Display bias analysis."""
        st.subheader("Bias Analysis")
        
        # Extract bias metrics
        bias_metrics = bias_results['bias_metrics']
        
        # Create radar chart for bias analysis
        categories = []
        bias_scores = []
        
        for bias_type, data in bias_metrics.items():
            categories.append(bias_type.title())
            bias_scores.append(data['score'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=bias_scores,
            theta=categories,
            fill='toself',
            name='Bias Score',
            line_color='darkblue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Bias Analysis (Lower = Better Balance)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display bias breakdown
        st.markdown("### Detailed Bias Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Political bias
            if 'political' in bias_metrics:
                political_data = bias_metrics['political']['percentages']
                labels = list(political_data.keys())
                values = list(political_data.values())
                
                fig = px.pie(
                    names=labels,
                    values=values,
                    title="Political Perspective Distribution",
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Perspective bias
            if 'perspective' in bias_metrics:
                perspective_data = bias_metrics['perspective']['percentages']
                labels = list(perspective_data.keys())
                values = list(perspective_data.values())
                
                fig = px.pie(
                    names=labels,
                    values=values,
                    title="Information Source Perspective",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Display sentiment breakdown
        if 'sentiment' in bias_metrics:
            sentiment_data = bias_metrics['sentiment']['percentages']
            labels = list(sentiment_data.keys())
            values = list(sentiment_data.values())
            
            fig = px.bar(
                x=labels,
                y=values,
                title="Content Sentiment Analysis",
                color=values,
                color_continuous_scale='RdBu',
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display source diversity
        st.markdown(f"### Source Diversity Score: {bias_results['domain_diversity']:.2f}")
        st.markdown("""
        * **0.0-0.3**: Very limited sources (echo chamber risk)
        * **0.3-0.6**: Moderate diversity of sources
        * **0.6-1.0**: High diversity of information sources
        """)
    
    def _display_topic_diversity(self, topic_results):
        """Display topic diversity analysis."""
        st.subheader("Topic Diversity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create gauge for diversity score
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=topic_results['topic_diversity_score'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Topic Diversity Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "royalblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightcoral"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': topic_results['topic_diversity_score'] * 100
                    }
                }
            ))
            
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation text
            diversity_level = "Low"
            if topic_results['topic_diversity_score'] > 0.7:
                diversity_level = "High"
            elif topic_results['topic_diversity_score'] > 0.3:
                diversity_level = "Moderate"
                
            st.markdown(f"**Diversity Level:** {diversity_level}")
            st.markdown(f"**Interpretation:** {topic_results.get('diversity_interpretation', 'This score represents how well-distributed your content is across different topics.')}")
        
        with col2:
            # Topic distribution chart
            if 'topic_distribution' in topic_results and len(topic_results['topic_distribution']) > 0:
                labels = [item[0] for item in topic_results['topic_distribution']]
                values = [item[1] for item in topic_results['topic_distribution']]
                
                pie_fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.4,
                    textinfo='label+percent',
                    insidetextorientation='radial'
                )])
                
                pie_fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3)
                )
                st.plotly_chart(pie_fig, use_container_width=True)
            else:
                st.info("No topic distribution data available.")
        
        # Display top topics with keywords
        if 'top_topics' in topic_results and len(topic_results['top_topics']) > 0:
            st.subheader("Top Topics and Keywords")
            for i, (topic_name, keywords, weight) in enumerate(topic_results['top_topics'][:5], 1):
                with st.expander(f"Topic {i}: {topic_name} ({weight:.1%})"):
                    st.write(f"**Keywords:** {', '.join(keywords)}")
                    
                    # Show recommendations if available
                    if 'topic_recommendations' in topic_results and topic_name in topic_results['topic_recommendations']:
                        st.write("**Recommendations:**")
                        st.write(topic_results['topic_recommendations'][topic_name])
        
        # Display diversity insights
        if 'diversity_insights' in topic_results:
            st.subheader("Content Diversity Insights")
            for insight in topic_results['diversity_insights']:
                st.info(insight)
                
        # Display any improvement recommendations
        if 'diversity_recommendations' in topic_results:
            st.subheader("Recommendations to Improve Topic Diversity")
            for i, rec in enumerate(topic_results['diversity_recommendations'], 1):
                st.write(f"{i}. {rec}")