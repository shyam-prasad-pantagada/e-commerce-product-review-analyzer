# app.py - E-commerce Sentiment Analysis with Product Selection & Visualization
import pandas as pd
import numpy as np
import re
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os
import random
from datetime import datetime

class EcommerceSentimentAnalyzer:
    def __init__(self, dataset_path='ecommerce_reviews.csv'):
        # Initialize word lists for sentiment analysis
        self.positive_words = {
            'good', 'excellent', 'amazing', 'great', 'happy', 'love', 'best', 'perfect',
            'recommend', 'awesome', 'fantastic', 'wonderful', 'outstanding', 'brilliant',
            'superb', 'nice', 'satisfied', 'pleased', 'delighted', 'impressed', 'worth',
            'exceeded', 'perfectly', 'perfect', 'lovely', 'enjoyed', 'pleasure', 'excellent'
        }
        self.negative_words = {
            'bad', 'poor', 'terrible', 'worst', 'disappointed', 'waste', 'horrible',
            'awful', 'broken', 'cheap', 'useless', 'junk', 'trash', 'garbage',
            'rubbish', 'dislike', 'hate', 'pathetic', 'fake', 'scam', 'regret',
            'return', 'refund', 'damaged', 'defective', 'slow', 'missing', 'broken'
        }
        self.neutral_words = {'okay', 'average', 'normal', 'fine', 'acceptable', 'mediocre', 'standard'}
        
        # Load dataset
        self.dataset = []
        self.products = []  # Unique products list
        self.product_reviews = {}  # Reviews grouped by product
        
        if os.path.exists(dataset_path):
            self.load_dataset(dataset_path)
            print(f"📊 Loaded dataset with {len(self.dataset)} reviews and {len(self.products)} products")
        else:
            self.create_sample_dataset()
            print("📝 Created sample dataset")
    
    def load_dataset(self, dataset_path):
        """Load e-commerce review dataset"""
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
                
                # Standardize column names
                column_map = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'product' in col_lower or 'item' in col_lower or 'name' in col_lower:
                        column_map[col] = 'product_name'
                    elif 'review' in col_lower or 'text' in col_lower or 'comment' in col_lower:
                        column_map[col] = 'review_text'
                    elif 'rating' in col_lower or 'star' in col_lower:
                        column_map[col] = 'rating'
                    elif 'category' in col_lower or 'type' in col_lower:
                        column_map[col] = 'category'
                
                if column_map:
                    df = df.rename(columns=column_map)
                
                # Ensure required columns exist
                if 'product_name' not in df.columns:
                    if len(df.columns) > 0:
                        df['product_name'] = df.iloc[:, 0]
                
                if 'review_text' not in df.columns:
                    if len(df.columns) > 1:
                        df['review_text'] = df.iloc[:, 1]
                    else:
                        df['review_text'] = 'No review text available'
                
                if 'rating' not in df.columns:
                    df['rating'] = 3  # Default rating
                
                if 'category' not in df.columns:
                    df['category'] = 'General'
                
                # Convert to records
                for _, row in df.iterrows():
                    try:
                        # Parse rating (1-5 scale)
                        rating = self.parse_rating(row.get('rating', 3))
                        review_text = str(row.get('review_text', ''))
                        
                        # Get sentiment from rating (1-2: negative, 3: neutral, 4-5: positive)
                        if rating >= 4:
                            sentiment = 'positive'
                        elif rating == 3:
                            sentiment = 'neutral'
                        else:
                            sentiment = 'negative'
                        
                        # Also analyze text sentiment for confirmation
                        text_sentiment = self.analyze_sentiment(review_text)
                        
                        # If rating and text sentiment disagree, use weighted average
                        if sentiment != text_sentiment:
                            # Give more weight to rating (60%) than text (40%)
                            if rating >= 4 or (rating == 3 and text_sentiment == 'neutral'):
                                sentiment = text_sentiment if random.random() < 0.4 else sentiment
                        
                        product_name = str(row.get('product_name', 'Unknown Product')).strip()
                        if not product_name:
                            product_name = 'Unknown Product'
                        
                        review = {
                            'product_name': product_name,
                            'review_text': review_text,
                            'rating': rating,
                            'category': str(row.get('category', 'General')),
                            'sentiment': sentiment,
                            'date': datetime.now().strftime('%Y-%m-%d')
                        }
                        
                        self.dataset.append(review)
                        
                        # Add to product list if not already there
                        if product_name not in self.products:
                            self.products.append(product_name)
                        
                        # Group reviews by product
                        if product_name not in self.product_reviews:
                            self.product_reviews[product_name] = []
                        self.product_reviews[product_name].append(review)
                        
                    except Exception as e:
                        print(f"Error processing row: {e}")
                        continue
                
                print(f"✅ Successfully loaded {len(self.dataset)} reviews")
                print(f"✅ Found {len(self.products)} unique products")
                
            else:
                print(f"❌ Unsupported file format: {dataset_path}")
                self.create_sample_dataset()
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            self.create_sample_dataset()
    
    def parse_rating(self, rating):
        """Parse rating to ensure it's between 1-5"""
        try:
            # Convert to float first
            if isinstance(rating, str):
                rating = rating.strip()
                # Extract first number from string
                match = re.search(r'(\d+(\.\d+)?)', rating)
                if match:
                    rating = float(match.group(1))
                else:
                    rating = 3.0
            else:
                rating = float(rating)
            
            # Ensure rating is between 1-5
            rating = max(1.0, min(5.0, rating))
            
            # Round to nearest integer for sentiment classification
            return int(round(rating))
            
        except:
            return 3  # Default to neutral
    
    def create_sample_dataset(self):
        """Create sample dataset with ratings 1-5"""
        sample_products = [
            "iPhone 15 Pro Max",
            "Samsung Galaxy S24 Ultra",
            "Sony WH-1000XM5 Headphones",
            "Nike Air Max 270",
            "Amazon Echo Dot 5th Gen",
            "Dell XPS 15 Laptop",
            "Instant Pot Duo Plus",
            "Kindle Paperwhite",
            "PlayStation 5 Console",
            "Bose QuietComfort Earbuds"
        ]
        
        sample_categories = ["Electronics", "Electronics", "Electronics", "Footwear", 
                           "Smart Home", "Computers", "Kitchen", "Electronics", 
                           "Gaming", "Electronics"]
        
        sample_reviews = [
            ("Excellent phone! Camera quality is amazing.", 5),
            ("Good battery life but expensive.", 4),
            ("Average performance, could be better.", 3),
            ("Disappointed with the sound quality.", 2),
            ("Worst purchase ever, stopped working in 2 days.", 1),
            ("Absolutely love it! Best in class.", 5),
            ("Okay product, nothing special.", 3),
            ("Very good value for money.", 4),
            ("Terrible customer service.", 1),
            ("Perfect for daily use.", 5),
            ("Not what I expected, returning it.", 2),
            ("Satisfactory but could improve.", 3),
            ("Amazing features, highly recommended!", 5),
            ("Poor build quality.", 2),
            ("Works fine for the price.", 4)
        ]
        
        for product_idx, product in enumerate(sample_products):
            category = sample_categories[product_idx % len(sample_categories)]
            
            # Create 5-10 reviews per product
            for i in range(random.randint(5, 10)):
                review_text, rating = random.choice(sample_reviews)
                rating = self.parse_rating(rating)
                
                # Determine sentiment from rating
                if rating >= 4:
                    sentiment = 'positive'
                elif rating == 3:
                    sentiment = 'neutral'
                else:
                    sentiment = 'negative'
                
                review = {
                    'product_name': product,
                    'review_text': f"Review {i+1}: {review_text}",
                    'rating': rating,
                    'category': category,
                    'sentiment': sentiment,
                    'date': datetime.now().strftime('%Y-%m-%d')
                }
                
                self.dataset.append(review)
                
                if product not in self.products:
                    self.products.append(product)
                
                if product not in self.product_reviews:
                    self.product_reviews[product] = []
                self.product_reviews[product].append(review)
        
        # Save sample dataset to CSV
        self.save_sample_dataset()
    
    def save_sample_dataset(self):
        """Save the sample dataset to CSV for future use"""
        df = pd.DataFrame(self.dataset)
        df.to_csv('ecommerce_reviews.csv', index=False)
        print("💾 Sample dataset saved to 'ecommerce_reviews.csv'")
    
    def analyze_sentiment(self, text):
        """Analyze sentiment from text (fallback method)"""
        if not isinstance(text, str):
            return 'neutral'
        
        text_lower = text.lower()
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        neu_count = sum(1 for word in self.neutral_words if word in text_lower)
        
        total = pos_count + neg_count + neu_count
        
        if total == 0:
            return 'neutral'
        
        if pos_count > neg_count and pos_count > neu_count:
            return 'positive'
        elif neg_count > pos_count and neg_count > neu_count:
            return 'negative'
        else:
            return 'neutral'
    
    def get_product_sentiment_stats(self, product_name):
        """Get sentiment statistics for a specific product"""
        if product_name not in self.product_reviews:
            return {
                'product_name': product_name,
                'total_reviews': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'avg_rating': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'reviews': []
            }
        
        reviews = self.product_reviews[product_name]
        total = len(reviews)
        
        positive = sum(1 for r in reviews if r['sentiment'] == 'positive')
        negative = sum(1 for r in reviews if r['sentiment'] == 'negative')
        neutral = sum(1 for r in reviews if r['sentiment'] == 'neutral')
        
        avg_rating = sum(r['rating'] for r in reviews) / total if total > 0 else 0
        
        return {
            'product_name': product_name,
            'total_reviews': total,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'avg_rating': round(avg_rating, 2),
            'sentiment_distribution': {
                'positive': positive,
                'negative': negative,
                'neutral': neutral
            },
            'reviews': reviews[:20]  # Return first 20 reviews
        }
    
    def get_all_products(self):
        """Get list of all unique products"""
        return sorted(self.products)
    
    def get_overall_stats(self):
        """Get overall statistics"""
        total_reviews = len(self.dataset)
        positive = sum(1 for r in self.dataset if r['sentiment'] == 'positive')
        negative = sum(1 for r in self.dataset if r['sentiment'] == 'negative')
        neutral = sum(1 for r in self.dataset if r['sentiment'] == 'neutral')
        
        avg_rating = sum(r['rating'] for r in self.dataset) / total_reviews if total_reviews > 0 else 0
        
        return {
            'total_reviews': total_reviews,
            'total_products': len(self.products),
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'avg_rating': round(avg_rating, 2),
            'sentiment_percentages': {
                'positive': round((positive / total_reviews) * 100, 2) if total_reviews > 0 else 0,
                'negative': round((negative / total_reviews) * 100, 2) if total_reviews > 0 else 0,
                'neutral': round((neutral / total_reviews) * 100, 2) if total_reviews > 0 else 0
            }
        }

# HTTP Server
class EcommerceHandler(BaseHTTPRequestHandler):
    analyzer = EcommerceSentimentAnalyzer()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        
        try:
            if path == '/products':
                # Get all products
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                products = self.analyzer.get_all_products()
                response = {
                    'success': True,
                    'products': products,
                    'count': len(products)
                }
                self.wfile.write(json.dumps(response).encode())
                
            elif path == '/product_stats':
                # Get stats for specific product
                query = parse_qs(parsed.query)
                product_name = query.get('product', [''])[0]
                
                if not product_name:
                    self.send_error(400, 'Product name is required')
                    return
                
                stats = self.analyzer.get_product_sentiment_stats(product_name)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    'success': True,
                    'stats': stats
                }
                self.wfile.write(json.dumps(response).encode())
                
            elif path == '/overall_stats':
                # Get overall statistics
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                stats = self.analyzer.get_overall_stats()
                response = {
                    'success': True,
                    'stats': stats
                }
                self.wfile.write(json.dumps(response).encode())
                
            elif path == '/dataset_info':
                # Get dataset information
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    'success': True,
                    'total_reviews': len(self.analyzer.dataset),
                    'total_products': len(self.analyzer.products),
                    'dataset_sample': self.analyzer.dataset[:5] if self.analyzer.dataset else []
                }
                self.wfile.write(json.dumps(response).encode())
                
            else:
                # Default response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    'success': True,
                    'message': 'E-commerce Sentiment Analysis API',
                    'endpoints': [
                        '/products - Get all products',
                        '/product_stats?product=NAME - Get product statistics',
                        '/overall_stats - Get overall statistics',
                        '/dataset_info - Get dataset information'
                    ]
                }
                self.wfile.write(json.dumps(response).encode())
                
        except Exception as e:
            self.send_error(500, f'Server error: {str(e)}')
    
    def do_POST(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/analyze_product':
            # Analyze sentiment for a product
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode())
                product_name = data.get('product_name', '')
                
                if not product_name:
                    self.send_error(400, 'Product name is required')
                    return
                
                stats = self.analyzer.get_product_sentiment_stats(product_name)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    'success': True,
                    'product_name': product_name,
                    'stats': stats
                }
                self.wfile.write(json.dumps(response).encode())
                
            except json.JSONDecodeError:
                self.send_error(400, 'Invalid JSON data')
            except Exception as e:
                self.send_error(500, f'Server error: {str(e)}')
        
        else:
            self.send_error(404, 'Endpoint not found')

def run_server(port=5000):
    server = HTTPServer(('localhost', port), EcommerceHandler)
    print(f"🚀 E-commerce Sentiment Analysis Server")
    print(f"🌐 Running on: http://localhost:{port}")
    print(f"📊 Dataset: {len(EcommerceHandler.analyzer.dataset)} reviews")
    print(f"🛍️ Products: {len(EcommerceHandler.analyzer.products)} unique products")
    print("=" * 50)
    print("📋 Available Endpoints:")
    print(f"  GET  http://localhost:{port}/products")
    print(f"  GET  http://localhost:{port}/product_stats?product=NAME")
    print(f"  GET  http://localhost:{port}/overall_stats")
    print(f"  POST http://localhost:{port}/analyze_product")
    print("=" * 50)
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        server.server_close()

if __name__ == '__main__':
    run_server()