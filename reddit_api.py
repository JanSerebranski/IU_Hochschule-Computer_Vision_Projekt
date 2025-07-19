import requests
import os
import json
from reddit_config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

class RedditAPI:
    def __init__(self):
        self.client_id = REDDIT_CLIENT_ID
        self.client_secret = REDDIT_CLIENT_SECRET
        self.user_agent = REDDIT_USER_AGENT
        self.base_url = 'https://www.reddit.com'
        self.headers = {'User-Agent': self.user_agent}

    def get_image_posts(self, limit=20):
        # Holt die neuesten Bild-Posts aus r/all
        url = f'{self.base_url}/r/all.json?limit={limit}'
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            return []
        posts = response.json().get('data', {}).get('children', [])
        image_posts = []
        for post in posts:
            data = post['data']
            if data.get('post_hint') == 'image' and data.get('url'):
                image_posts.append({
                    'id': data['id'],
                    'title': data.get('title'),
                    'image_url': data.get('url'),
                    'permalink': f"https://reddit.com{data.get('permalink')}",
                    'author': data.get('author'),
                    'created_utc': data.get('created_utc')
                })
        return image_posts

    def get_post_details(self, post_id):
        # Holt Details zu einem bestimmten Post
        url = f'{self.base_url}/by_id/t3_{post_id}.json'
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            return None
        posts = response.json().get('data', {}).get('children', [])
        if not posts:
            return None
        data = posts[0]['data']
        if data.get('post_hint') == 'image' and data.get('url'):
            return {
                'id': data['id'],
                'title': data.get('title'),
                'image_url': data.get('url'),
                'permalink': f"https://reddit.com{data.get('permalink')}",
                'author': data.get('author'),
                'created_utc': data.get('created_utc')
            }
        return None 