"""
Augment training data with plain benign GET requests.

Problem: Model learned that bare paths (no query, no body) are probes,
because Zenodo data dominates the bare-path samples.

Solution: Add more plain benign navigation patterns - the kind of traffic
real users generate when clicking around a website.
"""

import pandas as pd
import random
random.seed(42)

# Common legitimate web pages (without queries/bodies)
benign_pages = [
    "/", "/home", "/index.html", "/index.php",
    "/about", "/about-us", "/contact", "/contact-us", "/help", "/support",
    "/login", "/register", "/signup", "/sign-up", "/sign-in",
    "/logout", "/sign-out", "/account", "/profile", "/settings",
    "/dashboard", "/my-account", "/preferences",
    "/products", "/services", "/catalog", "/shop", "/store",
    "/cart", "/checkout", "/orders", "/wishlist",
    "/search", "/categories", "/blog", "/news",
    "/faq", "/terms", "/terms-of-service", "/privacy", "/privacy-policy",
    "/sitemap", "/sitemap.xml", "/robots.txt",
    
    # Juice Shop specific
    "/api/Challenges/", "/api/Products/", "/api/Feedbacks/", 
    "/api/Users/", "/api/BasketItems/", "/api/Quantitys/",
    "/rest/products", "/rest/products/search",
    "/rest/user/whoami", "/rest/admin/application-version",
    "/rest/languages", "/rest/memories", "/rest/chatbot/status",
    
    # Common subpages
    "/products/laptops", "/products/phones", "/products/electronics",
    "/blog/post/1", "/blog/post/2", "/blog/categories",
    "/articles", "/news/latest",
    
    # Account pages
    "/account/orders", "/account/addresses", "/account/payment-methods",
    "/profile/edit", "/profile/security", "/profile/notifications",
]

# Generate plain GET requests (no query string, no body)
new_samples = []
for page in benign_pages:
    # Add multiple variations to give weight
    for _ in range(20):
        new_samples.append({
            'method': 'GET',
            'path': page,
            'query_string': '',
            'body': '',
            'label': 'benign',
        })

# Common simple GET with single query param (still benign navigation)
for page in ['/products', '/blog', '/news', '/articles', '/categories', '/shop']:
    for sort in ['asc', 'desc', 'newest', 'oldest', 'popular']:
        for _ in range(3):
            new_samples.append({
                'method': 'GET',
                'path': page,
                'query_string': f'sort={sort}',
                'body': '',
                'label': 'benign',
            })
    for page_num in range(1, 11):
        new_samples.append({
            'method': 'GET',
            'path': page,
            'query_string': f'page={page_num}',
            'body': '',
            'label': 'benign',
        })

# Common ID-based navigation
import string
for prefix in ['/api/Products/', '/products/', '/articles/', '/users/', '/blog/post/']:
    for _ in range(20):
        item_id = random.randint(1, 1000)
        new_samples.append({
            'method': 'GET',
            'path': f'{prefix}{item_id}',
            'query_string': '',
            'body': '',
            'label': 'benign',
        })

# Search with normal queries
normal_searches = ['laptop', 'shoes', 'shirt', 'book', 'phone', 'music', 'tea', 'coffee',
                   'recipes', 'movies', 'travel', 'fitness', 'cooking']
for term in normal_searches:
    for path in ['/search', '/api/search', '/rest/products/search']:
        for _ in range(3):
            new_samples.append({
                'method': 'GET',
                'path': path,
                'query_string': f'q={term}',
                'body': '',
                'label': 'benign',
            })

# Login attempts (POST with normal credentials)
for _ in range(100):
    name = random.choice(['ali', 'sara', 'john', 'jane', 'ahmed', 'maria', 'tom', 'lisa'])
    domain = random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'example.com'])
    email = f"{name}{random.randint(1,99)}@{domain}"
    pwd = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    
    body = f'{{"email":"{email}","password":"{pwd}"}}'
    new_samples.append({
        'method': 'POST',
        'path': '/rest/user/login',
        'query_string': '',
        'body': body,
        'label': 'benign',
    })

print(f"Generated {len(new_samples)} new benign samples")

# Append to existing train.csv
train_df = pd.read_csv('unified_v2/train.csv').fillna("")
print(f"Original train.csv size: {len(train_df)}")

new_df = pd.DataFrame(new_samples)
augmented = pd.concat([train_df, new_df], ignore_index=True)

# Shuffle
augmented = augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# Save back
augmented.to_csv('unified_v2/train.csv', index=False)
print(f"New train.csv size: {len(augmented)}")
print(f"\nLabel distribution:")
print(augmented['label'].value_counts())

print(f"\nBare GET requests in benign (no qs, no body):")
benign = augmented[augmented['label'] == 'benign']
plain = benign[(benign['method']=='GET') & (benign['query_string']=='') & (benign['body']=='')]
print(f"  {len(plain)} (was 204)")
