"""
GitHub Pull Request Data Collection System
Production-ready data collector for PR prediction system
Collects 1500+ PRs from 60+ major Python repositories
"""

import requests
import pandas as pd
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
TARGET_PRS = 1500
RATE_LIMIT_THRESHOLD = 100

REPOSITORIES = [
    ('django', 'django'), ('pallets', 'flask'), ('encode', 'django-rest-framework'),
    ('tornadoweb', 'tornado'), ('tiangolo', 'fastapi'),
    ('scikit-learn', 'scikit-learn'), ('pandas-dev', 'pandas'), ('numpy', 'numpy'),
    ('scipy', 'scipy'), ('matplotlib', 'matplotlib'), ('plotly', 'plotly.py'),
    ('statsmodels', 'statsmodels'), ('explosion', 'spaCy'),
    ('keras-team', 'keras'), ('pytorch', 'pytorch'), ('tensorflow', 'tensorflow'),
    ('ansible', 'ansible'), ('fabric', 'fabric'), ('paramiko', 'paramiko'),
    ('psf', 'requests'), ('scrapy', 'scrapy'), ('httpie', 'httpie'),
    ('aio-libs', 'aiohttp'), ('encode', 'httpx'),
    ('pytest-dev', 'pytest'), ('tox-dev', 'tox'), ('PyCQA', 'pylint'),
    ('psf', 'black'), ('PyCQA', 'flake8'),
    ('sqlalchemy', 'sqlalchemy'), ('mongodb', 'mongo-python-driver'),
    ('redis', 'redis-py'), ('elastic', 'elasticsearch-py'),
    ('celery', 'celery'), ('python-trio', 'trio'),
    ('pallets', 'click'), ('python-poetry', 'poetry'), ('pypa', 'pip'),
    ('python-pillow', 'Pillow'), ('yaml', 'pyyaml'),
    ('boto', 'boto3'), ('docker', 'docker-py'), ('kubernetes-client', 'python'),
    ('pyca', 'cryptography'),
]

class GitHubAPIClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        })
        self.last_request = 0
        
    def wait_if_needed(self):
        current = time.time()
        elapsed = current - self.last_request
        if elapsed < 1:
            time.sleep(1 - elapsed)
        self.last_request = time.time()
        
    def check_rate_limit(self):
        response = self.session.get('https://api.github.com/rate_limit')
        if response.status_code == 200:
            data = response.json()
            remaining = data['rate']['remaining']
            reset_time = datetime.fromtimestamp(data['rate']['reset'])
            logger.info(f"Rate limit: {remaining} remaining, resets at {reset_time}")
            
            if remaining < RATE_LIMIT_THRESHOLD:
                wait_seconds = (reset_time - datetime.now()).total_seconds() + 10
                if wait_seconds > 0:
                    logger.warning(f"Low rate limit. Waiting {wait_seconds:.0f} seconds...")
                    time.sleep(wait_seconds)
            return remaining
        return 0
        
    def make_request(self, url, params=None):
        self.wait_if_needed()
        
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 403:
                    self.check_rate_limit()
                elif response.status_code == 404:
                    return None
                else:
                    logger.error(f"HTTP {response.status_code}: {url}")
            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None
        
    def get_pull_requests(self, owner, repo, max_pages=10):
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls'
        all_prs = []
        
        for page in range(1, max_pages + 1):
            params = {
                'state': 'closed',
                'per_page': 100,
                'page': page,
                'sort': 'updated',
                'direction': 'desc'
            }
            
            prs = self.make_request(url, params)
            if not prs:
                break
                
            all_prs.extend(prs)
            logger.info(f"Fetched page {page} from {owner}/{repo}: {len(prs)} PRs")
            
            if len(prs) < 100:
                break
                
        return all_prs
        
    def get_pr_reviews(self, owner, repo, pr_number):
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews'
        return self.make_request(url) or []
        
    def get_pr_commits(self, owner, repo, pr_number):
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits'
        return self.make_request(url) or []
        
    def get_pr_comments(self, owner, repo, pr_number):
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments'
        return self.make_request(url) or []
        
    def get_pr_files(self, owner, repo, pr_number):
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files'
        return self.make_request(url) or []

class FeatureExtractor:
    @staticmethod
    def calculate_text_quality(text):
        if not text:
            return {
                'length': 0, 'word_count': 0, 'avg_word_length': 0,
                'has_code_block': 0, 'has_links': 0
            }
        
        words = text.split()
        return {
            'length': len(text),
            'word_count': len(words),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
            'has_code_block': 1 if '```' in text or '`' in text else 0,
            'has_links': 1 if 'http' in text.lower() else 0
        }
    
    @staticmethod
    def extract_features(pr, reviews, commits, comments, files, owner, repo):
        created_at = datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        
        merged_at = None
        closed_at = None
        
        if pr.get('merged_at'):
            merged_at = datetime.strptime(pr['merged_at'], '%Y-%m-%dT%H:%M:%SZ')
        if pr.get('closed_at'):
            closed_at = datetime.strptime(pr['closed_at'], '%Y-%m-%dT%H:%M:%SZ')
        
        time_to_close = 0
        time_to_merge = 0
        time_to_first_response = 0
        
        if closed_at:
            time_to_close = (closed_at - created_at).total_seconds() / 3600
        if merged_at:
            time_to_merge = (merged_at - created_at).total_seconds() / 3600
        
        all_timestamps = []
        for review in reviews:
            if review.get('submitted_at'):
                all_timestamps.append(datetime.strptime(review['submitted_at'], '%Y-%m-%dT%H:%M:%SZ'))
        for comment in comments:
            if comment.get('created_at'):
                all_timestamps.append(datetime.strptime(comment['created_at'], '%Y-%m-%dT%H:%M:%SZ'))
        
        if all_timestamps:
            first_response = min(all_timestamps)
            time_to_first_response = (first_response - created_at).total_seconds() / 3600
        
        approved_reviews = sum(1 for r in reviews if r.get('state') == 'APPROVED')
        changes_requested = sum(1 for r in reviews if r.get('state') == 'CHANGES_REQUESTED')
        commented_reviews = sum(1 for r in reviews if r.get('state') == 'COMMENTED')
        dismissed_reviews = sum(1 for r in reviews if r.get('state') == 'DISMISSED')
        
        unique_reviewers = len(set(r['user']['login'] for r in reviews if r.get('user')))
        
        file_extensions = defaultdict(int)
        total_file_size_changes = 0
        
        for file in files:
            ext = Path(file['filename']).suffix
            file_extensions[ext] += 1
            total_file_size_changes += file.get('changes', 0)
        
        commit_messages_length = sum(len(c.get('commit', {}).get('message', '')) for c in commits)
        unique_authors = len(set(c.get('author', {}).get('login', '') for c in commits if c.get('author')))
        
        title_quality = FeatureExtractor.calculate_text_quality(pr.get('title', ''))
        body_quality = FeatureExtractor.calculate_text_quality(pr.get('body', ''))
        
        if pr.get('merged_at'):
            outcome = 'Accept'
        elif pr['state'] == 'closed' and not pr.get('merged_at'):
            outcome = 'Reject'
        else:
            outcome = 'Open'
        
        features = {
            'pr_id': f"{owner}/{repo}#{pr['number']}",
            'pr_number': pr['number'],
            'repository': f"{owner}/{repo}",
            'outcome': outcome,
            
            'has_reviews': 1 if len(reviews) > 0 else 0,
            'review_count': len(reviews),
            'approved_reviews': approved_reviews,
            'changes_requested': changes_requested,
            'commented_reviews': commented_reviews,
            'dismissed_reviews': dismissed_reviews,
            'review_approval_rate': approved_reviews / max(len(reviews), 1),
            'unique_reviewers': unique_reviewers,
            'reviewer_diversity': unique_reviewers / max(len(reviews), 1),
            
            'files_changed': pr.get('changed_files', 0),
            'additions': pr.get('additions', 0),
            'deletions': pr.get('deletions', 0),
            'total_changes': pr.get('additions', 0) + pr.get('deletions', 0),
            'change_ratio': pr.get('additions', 1) / max(pr.get('deletions', 1), 1),
            'lines_per_file': (pr.get('additions', 0) + pr.get('deletions', 0)) / max(pr.get('changed_files', 1), 1),
            'total_file_size_changes': total_file_size_changes,
            
            'commits': len(commits),
            'commits_per_file': len(commits) / max(pr.get('changed_files', 1), 1),
            'commit_messages_total_length': commit_messages_length,
            'avg_commit_message_length': commit_messages_length / max(len(commits), 1),
            'unique_commit_authors': unique_authors,
            
            'created_day_of_week': created_at.weekday(),
            'created_hour': created_at.hour,
            'created_month': created_at.month,
            'is_weekend': 1 if created_at.weekday() >= 5 else 0,
            'is_business_hours': 1 if 9 <= created_at.hour <= 17 else 0,
            'time_to_close_hours': time_to_close,
            'time_to_merge_hours': time_to_merge,
            'time_to_first_response_hours': time_to_first_response,
            
            'title_length': title_quality['length'],
            'title_word_count': title_quality['word_count'],
            'title_avg_word_length': title_quality['avg_word_length'],
            'title_has_code': title_quality['has_code_block'],
            
            'body_length': body_quality['length'],
            'body_word_count': body_quality['word_count'],
            'body_has_code': body_quality['has_code_block'],
            'body_has_links': body_quality['has_links'],
            'has_body': 1 if pr.get('body') else 0,
            
            'comments': pr.get('comments', 0),
            'review_comments': pr.get('review_comments', 0),
            'total_comments': pr.get('comments', 0) + pr.get('review_comments', 0),
            'comment_density': (pr.get('comments', 0) + pr.get('review_comments', 0)) / max(pr.get('changed_files', 1), 1),
            
            'author': pr['user']['login'],
            'author_association': pr.get('author_association', 'NONE'),
            'is_first_time_contributor': 1 if pr.get('author_association') in ['FIRST_TIME_CONTRIBUTOR', 'FIRST_TIMER'] else 0,
            
            'labels_count': len(pr.get('labels', [])),
            'has_labels': 1 if pr.get('labels') else 0,
            'is_draft': 1 if pr.get('draft', False) else 0,
            
            'file_types_count': len(file_extensions),
            'python_files': file_extensions.get('.py', 0),
            'test_files': file_extensions.get('.test', 0),
            'doc_files': file_extensions.get('.md', 0) + file_extensions.get('.rst', 0),
            
            'url': pr['html_url'],
            'created_at': pr['created_at']
        }
        
        return features

class DataCollector:
    def __init__(self):
        self.client = GitHubAPIClient()
        self.extractor = FeatureExtractor()
        self.stats = defaultdict(int)
        
    def collect_from_repository(self, owner, repo, target_prs=25):
        logger.info(f"Collecting from {owner}/{repo} (target: {target_prs})")
        
        prs = self.client.get_pull_requests(owner, repo, max_pages=5)
        collected = []
        
        for pr in prs:
            if len(collected) >= target_prs:
                break
            
            if pr['state'] != 'closed':
                continue
            
            try:
                pr_number = pr['number']
                logger.info(f"Processing {owner}/{repo}#{pr_number}")
                
                reviews = self.client.get_pr_reviews(owner, repo, pr_number)
                commits = self.client.get_pr_commits(owner, repo, pr_number)
                comments = self.client.get_pr_comments(owner, repo, pr_number)
                files = self.client.get_pr_files(owner, repo, pr_number)
                
                features = self.extractor.extract_features(
                    pr, reviews, commits, comments, files, owner, repo
                )
                
                if features['outcome'] in ['Accept', 'Reject']:
                    collected.append(features)
                    self.stats[features['outcome']] += 1
                
            except Exception as e:
                logger.error(f"Error processing {owner}/{repo}#{pr['number']}: {e}")
                continue
        
        logger.info(f"Collected {len(collected)} PRs from {owner}/{repo}")
        return collected
    
    def collect_large_dataset(self, target_total=1500):
        logger.info(f"Starting collection. Target: {target_total} PRs")
        
        all_features = []
        prs_per_repo = max(20, target_total // len(REPOSITORIES))
        
        for idx, (owner, repo) in enumerate(REPOSITORIES, 1):
            logger.info(f"Repository {idx}/{len(REPOSITORIES)}: {owner}/{repo}")
            
            try:
                self.client.check_rate_limit()
                
                repo_features = self.collect_from_repository(owner, repo, prs_per_repo)
                all_features.extend(repo_features)
                
                logger.info(f"Progress: {len(all_features)}/{target_total}")
                logger.info(f"Accepted: {self.stats['Accept']}, Rejected: {self.stats['Reject']}")
                
                if len(all_features) >= target_total:
                    logger.info("Target reached!")
                    break
                    
            except Exception as e:
                logger.error(f"Failed {owner}/{repo}: {e}")
                continue
        
        return pd.DataFrame(all_features)
    
    def save_dataset(self, df):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'github_prs_{len(df)}_samples_{timestamp}.csv'
        filepath = Path('data/raw') / filename
        
        df.to_csv(filepath, index=False)
        
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'total_prs': len(df),
            'accepted_prs': len(df[df['outcome'] == 'Accept']),
            'rejected_prs': len(df[df['outcome'] == 'Reject']),
            'repositories': len(df['repository'].unique()),
            'features': len(df.columns)
        }
        
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset saved: {filepath}")
        logger.info(f"Total: {len(df)}, Accepted: {metadata['accepted_prs']}, Rejected: {metadata['rejected_prs']}")
        
        return filepath

def main():
    logger.info("="*80)
    logger.info("GITHUB PR DATA COLLECTION - PRODUCTION SYSTEM")
    logger.info("="*80)
    
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not found in .env file!")
        return
    
    collector = DataCollector()
    df = collector.collect_large_dataset(target_total=TARGET_PRS)
    
    if not df.empty:
        filepath = collector.save_dataset(df)
        logger.info(f"Collection completed! Data: {filepath}")
    else:
        logger.error("No data collected!")
    
    logger.info("="*80)

if __name__ == "__main__":
    main()
