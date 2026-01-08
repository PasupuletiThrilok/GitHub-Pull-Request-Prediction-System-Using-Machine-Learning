#!/usr/bin/env python3
"""
Data Analyzer for Code Review Assistant Project
Analyzes collected GitHub pull request data
"""

import json
import os
import pandas as pd
from datetime import datetime
from collections import Counter
import glob

class DataAnalyzer:
    def __init__(self, data_path='data/raw/'):
        self.data_path = data_path
        self.all_data = []
        self.load_all_datasets()
    
    def load_all_datasets(self):
        """Load all JSON files from the data directory"""
        json_files = glob.glob(os.path.join(self.data_path, '*.json'))
        
        print(f"Found {len(json_files)} dataset files:")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.all_data.extend(data)
                    filename = os.path.basename(file_path)
                    print(f" {filename}: {len(data)} PRs")
            except Exception as e:
                print(f" Error loading {file_path}: {e}")
        
        print(f"\n Total PRs loaded: {len(self.all_data)}")
    
    def basic_statistics(self):
        """Show basic statistics about the collected data"""
        if not self.all_data:
            print(" No data loaded!")
            return
        
        print("\n" + "="*60)
        print(" BASIC STATISTICS")
        print("="*60)
        
        # Repository breakdown
        repos = Counter(pr['repo'] for pr in self.all_data)
        print(f" Repositories ({len(repos)}):")
        for repo, count in repos.items():
            print(f"   {repo}: {count} PRs")
        
        # PR states
        merged = sum(1 for pr in self.all_data if pr['merged_at'])
        closed = sum(1 for pr in self.all_data if pr['state'] == 'closed' and not pr['merged_at'])
        print(f"\n PR Status:")
        print(f"   Merged: {merged}")
        print(f"   Closed (not merged): {closed}")
        print(f"   Total: {len(self.all_data)}")
        
        # Review statistics
        with_reviews = [pr for pr in self.all_data if pr['reviews']]
        print(f"\n Reviews:")
        print(f"   PRs with reviews: {len(with_reviews)}")
        print(f"   PRs without reviews: {len(self.all_data) - len(with_reviews)}")
        
        # Code changes
        total_additions = sum(pr['additions'] for pr in self.all_data)
        total_deletions = sum(pr['deletions'] for pr in self.all_data)
        total_files = sum(pr['changed_files'] for pr in self.all_data)
        
        print(f"\n Code Changes:")
        print(f"   Total lines added: {total_additions:,}")
        print(f"   Total lines deleted: {total_deletions:,}")
        print(f"   Total files changed: {total_files:,}")
        print(f"   Average files per PR: {total_files/len(self.all_data):.1f}")
        
        # File type analysis
        all_files = []
        for pr in self.all_data:
            for file in pr['files']:
                all_files.append(file['filename'])
        
        file_extensions = Counter()
        for filename in all_files:
            ext = filename.split('.')[-1] if '.' in filename else 'no_ext'
            file_extensions[ext] += 1
        
        print(f"\n File Types (top 10):")
        for ext, count in file_extensions.most_common(10):
            print(f"   .{ext}: {count} files")
    
    def review_analysis(self):
        """Analyze review patterns"""
        print("\n" + "="*60)
        print(" REVIEW ANALYSIS")
        print("="*60)
        
        reviewed_prs = [pr for pr in self.all_data if pr['reviews']]
        
        if not reviewed_prs:
            print(" No PRs with reviews found!")
            return
        
        # Review states
        all_review_states = []
        for pr in reviewed_prs:
            for review in pr['reviews']:
                all_review_states.append(review['state'])
        
        review_counter = Counter(all_review_states)
        print(f" Review States:")
        for state, count in review_counter.items():
            print(f"   {state}: {count}")
        
        # Reviews per PR
        reviews_per_pr = [len(pr['reviews']) for pr in reviewed_prs]
        print(f"\n Reviews per PR:")
        print(f"   Average: {sum(reviews_per_pr)/len(reviews_per_pr):.1f}")
        print(f"   Max: {max(reviews_per_pr)}")
        print(f"   Min: {min(reviews_per_pr)}")
    
    def detailed_examples(self, num_examples=3):
        """Show detailed examples of collected data"""
        print("\n" + "="*60)
        print(f" DETAILED EXAMPLES (showing {num_examples})")
        print("="*60)
        
        # Show examples of different types
        examples = []
        
        # Find a merged PR with reviews
        merged_with_review = next((pr for pr in self.all_data 
                                 if pr['merged_at'] and pr['reviews']), None)
        if merged_with_review:
            examples.append(("Merged PR with Reviews", merged_with_review))
        
        # Find a closed PR with reviews
        closed_with_review = next((pr for pr in self.all_data 
                                 if not pr['merged_at'] and pr['reviews']), None)
        if closed_with_review:
            examples.append(("Closed PR with Reviews", closed_with_review))
        
        # Find a large PR (many file changes)
        large_pr = max(self.all_data, key=lambda x: x['changed_files'])
        examples.append(("Largest PR", large_pr))
        
        for i, (label, pr) in enumerate(examples[:num_examples]):
            print(f"\n🔸 {label}:")
            print(f"   PR #{pr['pr_number']}: {pr['title'][:60]}...")
            print(f"   Repository: {pr['repo']}")
            print(f"   Author: {pr['user']}")
            print(f"   State: {pr['state']} | Merged: {'Yes' if pr['merged_at'] else 'No'}")
            print(f"   Changes: +{pr['additions']} -{pr['deletions']} lines")
            print(f"   Files changed: {pr['changed_files']}")
            print(f"   Reviews: {len(pr['reviews'])}")
            
            if pr['reviews']:
                review_states = [r['state'] for r in pr['reviews']]
                print(f"   Review states: {', '.join(review_states)}")
            
            # Show Python files
            py_files = [f['filename'] for f in pr['files'] 
                       if f['filename'].endswith('.py')][:3]
            if py_files:
                print(f"   Python files: {', '.join(py_files)}")
    
    def data_quality_check(self):
        """Check data quality and completeness"""
        print("\n" + "="*60)
        print(" DATA QUALITY CHECK")
        print("="*60)
        
        # Check for missing fields
        required_fields = ['pr_number', 'title', 'state', 'user', 'reviews', 'files']
        
        for field in required_fields:
            missing_count = sum(1 for pr in self.all_data if not pr.get(field))
            if missing_count > 0:
                print(f"  {field}: {missing_count} PRs missing this field")
            else:
                print(f" {field}: Complete")
        
        # Check for empty reviews vs no reviews
        empty_reviews = sum(1 for pr in self.all_data if pr['reviews'] == [])
        none_reviews = sum(1 for pr in self.all_data if pr['reviews'] is None)
        print(f"\n Review data:")
        print(f"   Empty review arrays: {empty_reviews}")
        print(f"   None review fields: {none_reviews}")
        
        # Check file data
        prs_with_files = sum(1 for pr in self.all_data if pr['files'])
        print(f"   PRs with file data: {prs_with_files}/{len(self.all_data)}")
    
    def research_insights(self):
        """Provide insights relevant to research"""
        print("\n" + "="*60)
        print(" RESEARCH INSIGHTS")
        print("="*60)
        
        # Calculate review success rate
        reviewed_prs = [pr for pr in self.all_data if pr['reviews']]
        approved_prs = []
        rejected_prs = []
        
        for pr in reviewed_prs:
            review_states = [r['state'] for r in pr['reviews']]
            if 'APPROVED' in review_states:
                approved_prs.append(pr)
            elif 'REQUEST_CHANGES' in review_states:
                rejected_prs.append(pr)
        
        print(f" Review Outcomes:")
        print(f"   Approved: {len(approved_prs)}")
        print(f"   Changes requested: {len(rejected_prs)}")
        
        # Correlation between size and review
        small_prs = [pr for pr in self.all_data if pr['changed_files'] <= 3]
        large_prs = [pr for pr in self.all_data if pr['changed_files'] > 10]
        
        small_reviewed = sum(1 for pr in small_prs if pr['reviews'])
        large_reviewed = sum(1 for pr in large_prs if pr['reviews'])
        
        print(f"\n Size vs Review Pattern:")
        print(f"   Small PRs (≤3 files): {len(small_prs)}, reviewed: {small_reviewed}")
        print(f"   Large PRs (>10 files): {len(large_prs)}, reviewed: {large_reviewed}")
        
        # Machine learning potential
        ml_ready_prs = [pr for pr in self.all_data 
                       if pr['reviews'] and pr['files'] and pr['merged_at'] is not None]
        
        print(f"\n ML Training Potential:")
        print(f"   Complete PRs (reviews + files + outcome): {len(ml_ready_prs)}")
        print(f"   Percentage of total: {len(ml_ready_prs)/len(self.all_data)*100:.1f}%")

def main():
    """Main analysis function"""
    print(" GitHub Pull Request Prediction System Using ML - DATA ANALYSIS")
    print("="*60)
    
    analyzer = DataAnalyzer()
    
    if not analyzer.all_data:
        print(" No data found! Make sure you have JSON files in data/raw/")
        return
    
    # Run all analyses
    analyzer.basic_statistics()
    analyzer.review_analysis()
    analyzer.detailed_examples()
    analyzer.data_quality_check()
    analyzer.research_insights()
    
    print("\n" + "="*60)
    print(" NEXT STEPS:")
    print("="*60)
    print("1.  Data collection working!")
    print("2.  Collect more repositories for larger dataset")
    print("3.  Start feature extraction from code and reviews")
    print("4.  Begin building ML models")

if __name__ == "__main__":
    main()