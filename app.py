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
        self.positive_words = {'good', 'excellent', 'amazing', 'great', 'happy', 'love', 'best', 'perfect', 'recommend', 'awesome', 'fantastic', 'wonderful', 'outstanding', 'brilliant', 'superb', 'nice', 'satisfied', 'pleased', 'delighted', 'impressed', 'worth', 'exceeded', 'perfectly', 'lovely', 'enjoyed', 'pleasure'}
        self.negative_words = {'bad', 'poor', 'terrible', 'worst', 'disappointed', 'waste', 'horrible', 'awful', 'broken', 'cheap', 'useless', 'junk', 'trash', 'garbage', 'rubbish', 'dislike', 'hate', 'pathetic', 'fake', 'scam', 'regret', 'return', 'refund', 'damaged', 'defective', 'slow', 'missing'}
        self.neutral_words = {'okay', 'average', 'normal', 'fine', 'acceptable', 'mediocre', 'standard'}
        
        self.dataset = []
        self.products = []
        self.product_reviews = {}
        
        if os.path.exists(dataset_path):
            self.load_dataset(dataset_path)
            print(f"📊 Loaded dataset with {len(self.dataset)} reviews and {len(self.products)} products")
        else:
            self.create_sample_dataset()
            print("📝 Created sample dataset")

    def load_dataset(self, dataset_path):
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
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
                
                if 'product_name' not in df.columns and len(df.columns) > 0:
                    df['product_name'] = df.iloc[:, 0]
                if 'review_text' not in df.columns and len(df.columns) > 1:
                    df['review_text'] = df.iloc[:, 1]
                if 'rating' not in df.columns:
                    df['rating'] = 3
                if 'category' not in df.columns:
                    df['category'] = 'General'

                for _, row in df.iterrows():
                    try:
                        rating = self.parse_rating(row.get('rating', 3))
                        review_text = str(row.get('review_text', ''))
                        
                        if rating >= 4:
                            sentiment = 'positive'
                        elif rating == 3:
                            sentiment = 'neutral'
                        else:
                            sentiment = 'negative'
                        
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
                        if product_name not in self.products:
                            self.products.append(product_name)
                        if product_name not in self.product_reviews:
                            self.product_reviews[product_name] = []
                        self.product_reviews[product_name].append(review)
                    except Exception as e:
                        continue
        except Exception as e:
            self.create_sample_dataset()

    def parse_rating(self, rating):
        try:
            if isinstance(rating, str):
                match = re.search(r'(\d+(\.\d+)?)', rating.strip())
                rating = float(match.group(1)) if match else 3.0
            else:
                rating = float(rating)
            return int(round(max(1.0, min(5.0, rating))))
        except:
            return 3

    def create_sample_dataset(self):
        sample_products = ["iPhone 15 Pro Max", "Samsung Galaxy S24 Ultra", "Sony WH-1000XM5 Headphones", "Nike Air Max 270", "Amazon Echo Dot 5th Gen"]
        for product in sample_products:
            for i in range(5):
                review = {
                    'product_name': product,
                    'review_text': f"Sample review {i+1} for {product}",
                    'rating': 5,
                    'category': 'Electronics',
                    'sentiment': 'positive',
                    'date': datetime.now().strftime('%Y-%m-%d')
                }
                self.dataset.append(review)
                if product not in self.products:
                    self.products.append(product)
                if product not in self.product_reviews:
                    self.product_reviews[product] = []
                self.product_reviews[product].append(review)

    def get_product_sentiment_stats(self, product_name):
        reviews = self.product_reviews.get(product_name, [])
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
            'reviews': reviews[:20]
        }

    def get_all_products(self):
        return sorted(self.products)

    def get_overall_stats(self):
        total = len(self.dataset)
        positive = sum(1 for r in self.dataset if r['sentiment'] == 'positive')
        negative = sum(1 for r in self.dataset if r['sentiment'] == 'negative')
        neutral = sum(1 for r in self.dataset if r['sentiment'] == 'neutral')
        avg_rating = sum(r['rating'] for r in self.dataset) / total if total > 0 else 0
        return {
            'total_reviews': total,
            'total_products': len(self.products),
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'avg_rating': round(avg_rating, 2)
        }

class EcommerceHandler(BaseHTTPRequestHandler):
    analyzer = EcommerceSentimentAnalyzer()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, HEAD, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_HEAD(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == '/' or path == '':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
            elif path in ['/products', '/product_stats', '/overall_stats', '/dataset_info']:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
            else:
                self.send_error(404, 'Endpoint not found')
        except Exception as e:
            self.send_error(500, f'Server error: {str(e)}')

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        
        try:
            if path == '/' or path == '':
                try:
                    with open('index.html', 'rb') as f:
                        html_content = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(html_content)
                except FileNotFoundError:
                    self.send_response(404)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b"index.html not found in project root directory")
            elif path == '/products':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                products = self.analyzer.get_all_products()
                response = {'success': True, 'products': products, 'count': len(products)}
                self.wfile.write(json.dumps(response).encode())
            elif path == '/product_stats':
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
                self.wfile.write(json.dumps({'success': True, 'stats': stats}).encode())
            elif path == '/overall_stats':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                stats = self.analyzer.get_overall_stats()
                self.wfile.write(json.dumps({'success': True, 'stats': stats}).encode())
            else:
                self.send_error(404, 'Endpoint not found')
        except Exception as e:
                self.send_error(500, f'Server error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    server = HTTPServer(('0.0.0.0', port), EcommerceHandler)
    print(f"🚀 Server running on port {port}")
    server.serve_forever()
