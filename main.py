# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import hashlib
from collections import Counter
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, silhouette_score)

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier

# Utilities
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
# Load all datasets
print("="*60)
print("LOADING DATASETS")
print("="*60)

# Dataset 1: Email spam
emails_df = pd.read_csv('emails.csv')
print(f"\n1. Emails Dataset: {emails_df.shape}")
print(f"   Columns: {emails_df.columns.tolist()}")

# Dataset 2: SMS spam
# sms_df = pd.read_csv('sms_spam.csv')
# print(f"\n2. SMS Dataset: {sms_df.shape}")
# print(f"   Columns: {sms_df.columns.tolist()}")
try:
    sms_df = pd.read_csv('sms_spam.csv', sep='\t', header=None, encoding='latin-1', names=['label', 'message'])
    print(f"\n✓ SMS Dataset loaded: {sms_df.shape}")
    print(f"  Columns: {sms_df.columns.tolist()}")
except Exception as e:
    print(f"\n✗ Error loading sms_spam.csv: {e}")
    sms_df = pd.DataFrame()

# Dataset 3: Phishing emails
phishing_df = pd.read_csv('phishing_emails.csv')
print(f"\n3. Phishing Emails Dataset: {phishing_df.shape}")
print(f"   Columns: {phishing_df.columns.tolist()}")

# Dataset 4: Malware processes
malware_df = pd.read_csv('malware_dataset.csv')
print(f"\n4. Malware Dataset: {malware_df.shape}")
print(f"   Columns: {malware_df.columns.tolist()}")

print("\n" + "="*60)
print("\n" + "="*60)
print("DATA CLEANING - TEXT DATASETS")
print("="*60)

# Function to clean text
def clean_text(text):
    """
    Clean and normalize text data
    - Convert to lowercase
    - Remove URLs
    - Remove email addresses
    - Remove special characters
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Clean emails dataset
print("\n1. Cleaning Emails Dataset...")
emails_df['text_clean'] = emails_df['text'].apply(clean_text)
emails_df = emails_df.dropna(subset=['text_clean'])
emails_df = emails_df[emails_df['text_clean'].str.len() > 0]
print(f"   After cleaning: {emails_df.shape[0]} rows")
print(f"   Missing values: {emails_df.isnull().sum().sum()}")

# Clean SMS dataset
print("\n2. Cleaning SMS Dataset...")
if not sms_df.empty:
    sms_df['message_clean'] = sms_df['message'].apply(clean_text)
    sms_df = sms_df.dropna(subset=['message_clean'])
    sms_df = sms_df[sms_df['message_clean'].str.len() > 0]
    
    # Convert label to binary (spam=1, ham=0)
    sms_df['spam'] = sms_df['label'].map({'spam': 1, 'ham': 0})
    
    print(f"   After cleaning: {sms_df.shape[0]} rows")
    print(f"   Label distribution: {sms_df['spam'].value_counts().to_dict()}")
    print(f"   Missing values: {sms_df.isnull().sum().sum()}")
else:
    print("   Skipped - dataset not loaded")

# Clean phishing emails dataset
print("\n3. Cleaning Phishing Emails Dataset...")
phishing_df['Email Text_clean'] = phishing_df['Email Text'].apply(clean_text)
phishing_df = phishing_df.dropna(subset=['Email Text_clean'])
phishing_df = phishing_df[phishing_df['Email Text_clean'].str.len() > 0]
print(f"   After cleaning: {phishing_df.shape[0]} rows")
print(f"   Missing values: {phishing_df.isnull().sum().sum()}")

print("\n" + "="*60)
print("\n" + "="*60)
print("DATA CLEANING - MALWARE DATASET")
print("\n" + "="*60)
if not malware_df.empty:
    # Check missing values
    print("\nMissing values before cleaning:")
    missing_count = malware_df.isnull().sum().sum()
    if missing_count > 0:
        print(malware_df.isnull().sum()[malware_df.isnull().sum() > 0])
    else:
        print("No missing values")
    
    # Handle missing values
    numerical_cols = malware_df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if malware_df[col].isnull().sum() > 0:
            median_val = malware_df[col].median()
            malware_df[col].fillna(median_val, inplace=True)
            print(f"Filled {col} with median: {median_val}")
    
    # DON'T remove duplicates - they're valid process snapshots
    print(f"\nNote: Dataset contains {malware_df.duplicated(subset=['hash']).sum()} duplicate hashes")
    print("Keeping all rows as they represent different process states")
    
    # DON'T remove outliers - important for malware detection
    print("\nSkipping outlier removal to preserve malware patterns")
    
    print(f"\nFinal malware dataset shape: {malware_df.shape}")
    print(f"Missing values after cleaning: {malware_df.isnull().sum().sum()}")
else:
    print("   Skipped - dataset not loaded")

print("\n" + "="*60)
print("\n" + "="*60)
print("FEATURE ENGINEERING - TEXT DATA")
print("="*60)

# Define keyword dictionaries for multi-class classification
phishing_keywords = [
    'verify', 'account', 'suspended', 'login', 'password',
    'confirm', 'security', 'urgent', 'click here', 'update',
    'bank', 'paypal', 'credential', 'expire', 'validation'
]

promotion_keywords = [
    'free', 'win', 'winner', 'prize', 'offer', 'discount',
    'sale', 'limited time', 'buy now', 'deal', 'save',
    'congratulations', 'claim', 'bonus', 'gift', 'reward'
]

scam_keywords = [
    'lottery', 'inheritance', 'transfer', 'western union',
    'nigeria', 'prince', 'million', 'beneficiary', 'fund',
    'lawyer', 'diplomat', 'courier', 'tax refund', 'government grant'
]

def extract_text_features(text):
    """
    Extract comprehensive features from text
    """
    if not isinstance(text, str) or len(text) == 0:
        return {
            'length': 0, 'word_count': 0, 'url_count': 0,
            'caps_ratio': 0, 'exclamation_count': 0, 'question_count': 0,
            'digit_ratio': 0, 'special_char_count': 0,
            'phishing_kw_count': 0, 'promotion_kw_count': 0, 'scam_kw_count': 0
        }
    
    text_lower = text.lower()
    
    # Basic features
    length = len(text)
    word_count = len(text.split())
    
    # URL and email features
    url_count = len(re.findall(r'http|www', text_lower))
    email_count = len(re.findall(r'\S+@\S+', text_lower))
    
    # Text pattern features
    caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
    caps_ratio = caps_words / word_count if word_count > 0 else 0
    
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Character analysis
    digit_count = sum(c.isdigit() for c in text)
    digit_ratio = digit_count / length if length > 0 else 0
    
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    
    # Keyword counts
    phishing_kw = sum(1 for kw in phishing_keywords if kw in text_lower)
    promotion_kw = sum(1 for kw in promotion_keywords if kw in text_lower)
    scam_kw = sum(1 for kw in scam_keywords if kw in text_lower)
    
    return {
        'length': length,
        'word_count': word_count,
        'url_count': url_count,
        'email_count': email_count,
        'caps_ratio': caps_ratio,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'digit_ratio': digit_ratio,
        'special_char_count': special_chars,
        'phishing_kw_count': phishing_kw,
        'promotion_kw_count': promotion_kw,
        'scam_kw_count': scam_kw
    }

# Apply feature extraction to emails
print("\n1. Extracting features from emails...")
email_features = emails_df['text_clean'].apply(extract_text_features).apply(pd.Series)
emails_df = pd.concat([emails_df, email_features], axis=1)
print(f"   Features extracted: {email_features.columns.tolist()}")

# Apply feature extraction to SMS
print("\n2. Extracting features from SMS...")
sms_features = sms_df['message_clean'].apply(extract_text_features).apply(pd.Series)
sms_df = pd.concat([sms_df, sms_features], axis=1)

# Apply feature extraction to phishing emails
print("\n3. Extracting features from phishing emails...")
phishing_features = phishing_df['Email Text_clean'].apply(extract_text_features).apply(pd.Series)
phishing_df = pd.concat([phishing_df, phishing_features], axis=1)

print("\n" + "="*60)
print("\n" + "="*60)
print("CREATING MULTI-CLASS LABELS")
print("="*60)

def categorize_message(row):
    """
    Categorize messages into: Safe, Phishing, Promotion, Scam
    Based on keyword counts and patterns
    """
    phish_score = row['phishing_kw_count']
    promo_score = row['promotion_kw_count']
    scam_score = row['scam_kw_count']
    
    # If it's from phishing dataset and marked as phishing
    if 'Email Type' in row.index and 'phishing' in str(row['Email Type']).lower():
        return 'Phishing'
    
    # If no spam indicators
    if phish_score == 0 and promo_score == 0 and scam_score == 0:
        if 'spam' in row.index and row['spam'] == 0:
            return 'Safe'
        elif 'Email Type' in row.index and 'safe' in str(row['Email Type']).lower():
            return 'Safe'
    
    # Determine category based on highest score
    max_score = max(phish_score, promo_score, scam_score)
    
    if max_score == 0:
        return 'Safe'
    elif phish_score == max_score:
        return 'Phishing'
    elif promo_score == max_score:
        return 'Promotion'
    else:
        return 'Scam'

# Apply categorization
print("\n1. Categorizing email messages...")
emails_df['category'] = emails_df.apply(categorize_message, axis=1)

print("\n2. Categorizing SMS messages...")
sms_df['category'] = sms_df.apply(categorize_message, axis=1)

print("\n3. Categorizing phishing emails...")
phishing_df['category'] = phishing_df.apply(categorize_message, axis=1)

# Show distribution
print("\nEmail category distribution:")
print(emails_df['category'].value_counts())

print("\nSMS category distribution:")
print(sms_df['category'].value_counts())

print("\nPhishing emails category distribution:")
print(phishing_df['category'].value_counts())

print("\n" + "="*60)
print("\n" + "="*60)
print("COMBINING TEXT DATASETS")
print("\n" + "="*60)
all_text_df = pd.DataFrame()

# Define feature columns list
feature_cols = ['length', 'word_count', 'url_count', 'email_count', 'caps_ratio',
               'exclamation_count', 'question_count', 'digit_ratio', 'special_char_count',
               'phishing_kw_count', 'promotion_kw_count', 'scam_kw_count']

if not emails_df.empty:
    emails_combined = emails_df[['text_clean', 'spam', 'category'] + feature_cols].copy()
    emails_combined['source'] = 'email'
    all_text_df = pd.concat([all_text_df, emails_combined], ignore_index=True)

if not sms_df.empty and 'spam' in sms_df.columns:
    # Select only the columns we need
    sms_cols_needed = ['message_clean', 'spam', 'category'] + feature_cols
    sms_combined = sms_df[sms_cols_needed].copy()
    
    # Rename message_clean to text_clean
    sms_combined = sms_combined.rename(columns={'message_clean': 'text_clean'})
    sms_combined['source'] = 'sms'
    
    all_text_df = pd.concat([all_text_df, sms_combined], ignore_index=True)

if not phishing_df.empty:
    phishing_combined = phishing_df[['Email Text_clean', 'category'] + feature_cols].copy()
    phishing_combined = phishing_combined.rename(columns={'Email Text_clean': 'text_clean'})
    phishing_combined['spam'] = phishing_combined['category'].apply(lambda x: 0 if x == 'Safe' else 1)
    phishing_combined['source'] = 'phishing_email'
    
    all_text_df = pd.concat([all_text_df, phishing_combined], ignore_index=True)

print(f"   ✓ Combined text dataset: {all_text_df.shape}")
if not all_text_df.empty:
    print(f"   Source distribution:")
    print(all_text_df['source'].value_counts())
    print(f"\n   Category distribution:")
    print(all_text_df['category'].value_counts())

print("\n" + "="*80)
print("\n" + "="*60)
print("NORMALIZING MALWARE DATASET")
print("="*60)

# Select numerical features for normalization
feature_cols = [col for col in malware_df.columns 
                if col not in ['hash', 'classification', 'millisecond']]

print(f"\nFeatures to normalize: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:5]}")

# Check data types
print("\nData types:")
print(malware_df[feature_cols].dtypes.value_counts())

# Initialize scaler
scaler_malware = StandardScaler()

# Fit and transform
malware_features_scaled = scaler_malware.fit_transform(malware_df[feature_cols])

# Create scaled dataframe
malware_scaled_df = pd.DataFrame(
    malware_features_scaled,
    columns=feature_cols,
    index=malware_df.index
)

# Add back non-numerical columns
malware_scaled_df['hash'] = malware_df['hash'].values
malware_scaled_df['classification'] = malware_df['classification'].values

print(f"\nScaled malware dataset shape: {malware_scaled_df.shape}")
print("\nSample statistics after scaling:")
print(malware_scaled_df[feature_cols[:5]].describe())

print("\n" + "="*60)
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)
# Set up the plotting area
fig = plt.figure(figsize=(20, 12))

# 1. Spam distribution across all text sources
plt.subplot(3, 4, 1)
spam_dist = all_text_df['spam'].value_counts()
plt.pie(spam_dist.values, labels=['Ham', 'Spam'], autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'], startangle=90)
plt.title('Overall Spam Distribution', fontsize=14, fontweight='bold')

# 2. Category distribution
plt.subplot(3, 4, 2)
category_counts = all_text_df['category'].value_counts()
plt.barh(category_counts.index, category_counts.values, color='skyblue')
plt.xlabel('Count')
plt.title('Message Category Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# 3. Source distribution
plt.subplot(3, 4, 3)
source_counts = all_text_df['source'].value_counts()
colors_source = ['#3498db', '#e67e22', '#9b59b6']
plt.bar(source_counts.index, source_counts.values, color=colors_source)
plt.xlabel('Source')
plt.ylabel('Count')
plt.title('Messages by Source', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# 4. Message length distribution by spam/ham
plt.subplot(3, 4, 4)
spam_lengths = all_text_df[all_text_df['spam']==1]['length']
ham_lengths = all_text_df[all_text_df['spam']==0]['length']
plt.hist([ham_lengths, spam_lengths], bins=50, label=['Ham', 'Spam'],
         color=['#2ecc71', '#e74c3c'], alpha=0.7)
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.title('Message Length Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 5. URL count by category
plt.subplot(3, 4, 5)
url_by_cat = all_text_df.groupby('category')['url_count'].mean()
plt.bar(url_by_cat.index, url_by_cat.values, color='coral')
plt.xlabel('Category')
plt.ylabel('Average URL Count')
plt.title('Average URLs per Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# 6. Keyword counts by category
plt.subplot(3, 4, 6)
keyword_data = all_text_df.groupby('category')[
    ['phishing_kw_count', 'promotion_kw_count', 'scam_kw_count']
].mean()
keyword_data.plot(kind='bar', ax=plt.gca(), width=0.8)
plt.xlabel('Category')
plt.ylabel('Average Keyword Count')
plt.title('Keyword Patterns by Category', fontsize=14, fontweight='bold')
plt.legend(['Phishing', 'Promotion', 'Scam'], loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# 7. CAPS ratio by spam/ham
plt.subplot(3, 4, 7)
spam_caps = all_text_df[all_text_df['spam']==1]['caps_ratio']
ham_caps = all_text_df[all_text_df['spam']==0]['caps_ratio']
plt.boxplot([ham_caps, spam_caps], labels=['Ham', 'Spam'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'))
plt.ylabel('CAPS Ratio')
plt.title('CAPS Usage: Ham vs Spam', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 8. Exclamation marks by category
plt.subplot(3, 4, 8)
exclaim_by_cat = all_text_df.groupby('category')['exclamation_count'].mean()
plt.bar(exclaim_by_cat.index, exclaim_by_cat.values, color='gold')
plt.xlabel('Category')
plt.ylabel('Average Exclamation Count')
plt.title('Exclamation Marks by Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# 9. Correlation heatmap of text features
plt.subplot(3, 4, 9)
feature_cols_text = ['length', 'word_count', 'url_count', 'caps_ratio',
                     'exclamation_count', 'phishing_kw_count', 
                     'promotion_kw_count', 'scam_kw_count']
correlation_matrix = all_text_df[feature_cols_text].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)

# 10. Word count distribution by category
plt.subplot(3, 4, 10)
for category in all_text_df['category'].unique():
    data = all_text_df[all_text_df['category']==category]['word_count']
    plt.hist(data, bins=30, alpha=0.5, label=category)
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Word Count by Category', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 11. Special character usage
plt.subplot(3, 4, 11)
special_by_spam = all_text_df.groupby('spam')['special_char_count'].mean()
plt.bar(['Ham', 'Spam'], special_by_spam.values, 
        color=['#2ecc71', '#e74c3c'])
plt.ylabel('Average Special Characters')
plt.title('Special Character Usage', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 12. Digit ratio comparison
plt.subplot(3, 4, 12)
digit_by_cat = all_text_df.groupby('category')['digit_ratio'].mean()
plt.barh(digit_by_cat.index, digit_by_cat.values, color='lightgreen')
plt.xlabel('Average Digit Ratio')
plt.title('Digit Usage by Category', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('text_data_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nText data analysis visualization saved as 'text_data_analysis.png'")
# Malware dataset analysis
fig = plt.figure(figsize=(20, 10))

# 1. Classification distribution
plt.subplot(2, 4, 1)
class_dist = malware_df['classification'].value_counts()
plt.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%',
        startangle=90)
plt.title('Malware Classification Distribution', fontsize=14, fontweight='bold')

# 2. Top 10 important features - variance analysis
plt.subplot(2, 4, 2)
feature_variance = malware_scaled_df[feature_cols].var().sort_values(ascending=False)[:10]
plt.barh(range(len(feature_variance)), feature_variance.values, color='teal')
plt.yticks(range(len(feature_variance)), feature_variance.index, fontsize=9)
plt.xlabel('Variance')
plt.title('Top 10 Features by Variance', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# 3. Distribution of key features
plt.subplot(2, 4, 3)
key_features = ['total_vm', 'map_count', 'task_size', 'utime']
for feat in key_features:
    if feat in malware_df.columns:
        plt.hist(malware_df[feat], bins=50, alpha=0.5, label=feat)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Key Feature Distributions', fontsize=14, fontweight='bold')
plt.legend(fontsize=8)
plt.yscale('log')
plt.grid(axis='y', alpha=0.3)

# 4. Feature correlation heatmap (sample)
plt.subplot(2, 4, 4)
sample_features = ['state', 'prio', 'total_vm', 'map_count', 
                   'utime', 'stime', 'min_flt', 'maj_flt']
sample_features = [f for f in sample_features if f in malware_df.columns]
corr_matrix = malware_df[sample_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
            square=True, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation (Sample)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)

# 5. Box plot for normalized features
plt.subplot(2, 4, 5)
sample_scaled = malware_scaled_df[sample_features[:4]].sample(1000)
sample_scaled.boxplot()
plt.ylabel('Scaled Value')
plt.title('Normalized Feature Distributions', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# 6. Scatter plot: total_vm vs map_count
plt.subplot(2, 4, 6)
sample_malware = malware_df.sample(min(5000, len(malware_df)))
plt.scatter(sample_malware['total_vm'], sample_malware['map_count'],
           alpha=0.3, c=sample_malware['classification'].astype('category').cat.codes,
           cmap='viridis', s=10)
plt.xlabel('Total VM')
plt.ylabel('Map Count')
plt.title('Total VM vs Map Count', fontsize=14, fontweight='bold')
plt.colorbar(label='Classification')
plt.grid(alpha=0.3)

# 7. State distribution
plt.subplot(2, 4, 7)
state_counts = malware_df['state'].value_counts()[:10]
plt.bar(state_counts.index.astype(str), state_counts.values, color='orange')
plt.xlabel('State')
plt.ylabel('Count')
plt.title('Top 10 Process States', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# 8. Time-based features
plt.subplot(2, 4, 8)
time_features = ['utime', 'stime', 'gtime', 'cgtime']
time_features = [f for f in time_features if f in malware_df.columns]
time_data = malware_df[time_features].mean()
plt.bar(time_data.index, time_data.values, color='purple')
plt.xlabel('Time Feature')
plt.ylabel('Average Value')
plt.title('Average Time Features', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('malware_data_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nMalware data analysis visualization saved as 'malware_data_analysis.png'")
print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)

print("\n1. TEXT DATA STATISTICS")
print("-" * 60)
print("\nOverall Statistics:")
print(all_text_df[['length', 'word_count', 'url_count', 'caps_ratio',
                    'exclamation_count', 'phishing_kw_count',
                    'promotion_kw_count', 'scam_kw_count']].describe())

print("\n\nStatistics by Category:")
category_stats = all_text_df.groupby('category')[
    ['length', 'word_count', 'url_count', 'caps_ratio']
].agg(['mean', 'std', 'median'])
print(category_stats)

print("\n\n2. MALWARE DATA STATISTICS")
print("-" * 60)
print("\nKey Features Statistics:")
print(malware_df[sample_features].describe())
print("\n\nClass Distribution:")
print(malware_df['classification'].value_counts())

print("\n" + "="*60)
print("\n" + "="*60)
print("PREPARING DATA FOR MODEL TRAINING")
print("="*60)

# ============================================
# DATASET 1: Binary Spam Classification
# ============================================
print("\n1. Preparing Binary Spam Dataset (Model A - Naive Bayes)")
print("-" * 60)

# Combine email and SMS for binary classification
binary_spam_df = all_text_df[all_text_df['source'].isin(['email', 'sms'])].copy()

# TF-IDF Vectorization
tfidf_binary = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words='english'
)

X_binary_tfidf = tfidf_binary.fit_transform(binary_spam_df['text_clean'])
y_binary = binary_spam_df['spam'].values

print(f"Binary dataset shape: {X_binary_tfidf.shape}")
print(f"Class distribution: {Counter(y_binary)}")

# Train-test split
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    X_binary_tfidf, y_binary, test_size=0.25, random_state=42, stratify=y_binary
)

print(f"Training set: {X_train_binary.shape}")
print(f"Test set: {X_test_binary.shape}")

# ============================================
# DATASET 2: Multi-class Message Type
# ============================================
print("\n2. Preparing Multi-class Dataset (Model B - Random Forest)")
print("-" * 60)

# Use all text data for multi-class
multiclass_df = all_text_df.copy()

# Encode labels
label_encoder = LabelEncoder()
y_multiclass_encoded = label_encoder.fit_transform(multiclass_df['category'])

print(f"Classes: {label_encoder.classes_}")
print(f"Class distribution: {Counter(y_multiclass_encoded)}")

# Combine TF-IDF with manual features
tfidf_multi = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    stop_words='english'
)

X_multi_tfidf = tfidf_multi.fit_transform(multiclass_df['text_clean'])

# Manual features
manual_features = multiclass_df[[
    'length', 'word_count', 'url_count', 'caps_ratio',
    'exclamation_count', 'question_count', 'digit_ratio',
    'special_char_count', 'phishing_kw_count', 
    'promotion_kw_count', 'scam_kw_count'
]].values

# Scale manual features
scaler_multi = StandardScaler()
manual_features_scaled = scaler_multi.fit_transform(manual_features)

# Combine sparse and dense features
from scipy.sparse import hstack
X_multiclass_combined = hstack([X_multi_tfidf, manual_features_scaled])

print(f"Multi-class dataset shape: {X_multiclass_combined.shape}")

# Train-test split
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multiclass_combined, y_multiclass_encoded, 
    test_size=0.25, random_state=42, stratify=y_multiclass_encoded
)

print(f"Training set: {X_train_multi.shape}")
print(f"Test set: {X_test_multi.shape}")

# ============================================
# DATASET 3: Malware Classification
# ============================================
print("\n3. Preparing Malware Dataset (Model C - XGBoost)")
print("-" * 60)

# Features and labels
X_malware = malware_scaled_df[feature_cols].values
y_malware = malware_scaled_df['classification'].values

# Encode labels if categorical
if y_malware.dtype == 'object':
    label_encoder_malware = LabelEncoder()
    y_malware = label_encoder_malware.fit_transform(y_malware)
    print(f"Malware classes: {label_encoder_malware.classes_}")

print(f"Malware dataset shape: {X_malware.shape}")
print(f"Class distribution: {Counter(y_malware)}")

# Train-test split
X_train_malware, X_test_malware, y_train_malware, y_test_malware = train_test_split(
    X_malware, y_malware, test_size=0.25, random_state=42, stratify=y_malware
)

print(f"Training set: {X_train_malware.shape}")
print(f"Test set: {X_test_malware.shape}")

# ============================================
# DATASET 4: Clustering Data
# ============================================
print("\n4. Preparing Clustering Dataset (Model D - DBSCAN)")
print("-" * 60)

# Use TF-IDF from all text for clustering
X_clustering = tfidf_multi.transform(all_text_df['text_clean'])

print(f"Clustering dataset shape: {X_clustering.shape}")

print("\n" + "="*60)
print("\n" + "="*60)
print("MODEL A: NAIVE BAYES - BINARY SPAM CLASSIFICATION")
print("="*60)

# Train Naive Bayes
print("\nTraining Multinomial Naive Bayes...")
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_binary, y_train_binary)

# Predictions
y_pred_nb = nb_model.predict(X_test_binary)
y_pred_nb_proba = nb_model.predict_proba(X_test_binary)

# Evaluation
print("\n" + "="*60)
print("MODEL A EVALUATION")
print("="*60)

print("\nConfusion Matrix:")
cm_nb = confusion_matrix(y_test_binary, y_pred_nb)
print(cm_nb)

print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_nb, 
                          target_names=['Ham', 'Spam'], digits=4))

# Metrics
accuracy_nb = accuracy_score(y_test_binary, y_pred_nb)
precision_nb = precision_score(y_test_binary, y_pred_nb)
recall_nb = recall_score(y_test_binary, y_pred_nb)
f1_nb = f1_score(y_test_binary, y_pred_nb)
roc_auc_nb = roc_auc_score(y_test_binary, y_pred_nb_proba[:, 1])

print(f"\nAccuracy:  {accuracy_nb:.4f}")
print(f"Precision: {precision_nb:.4f}")
print(f"Recall:    {recall_nb:.4f}")
print(f"F1-Score:  {f1_nb:.4f}")
print(f"ROC-AUC:   {roc_auc_nb:.4f}")

# Cross-validation
print("\nCross-Validation Scores (5-fold):")
cv_scores_nb = cross_val_score(nb_model, X_train_binary, y_train_binary, 
                                cv=5, scoring='f1')
print(f"F1 Scores: {cv_scores_nb}")
print(f"Mean F1: {cv_scores_nb.mean():.4f} (+/- {cv_scores_nb.std():.4f})")

# Save model
import pickle
with open('model_a_naive_bayes.pkl', 'wb') as f:
    pickle.dump(nb_model, f)
with open('tfidf_binary.pkl', 'wb') as f:
    pickle.dump(tfidf_binary, f)

print("\nModel A saved successfully!")
print("="*60)
print("\n" + "="*60)
print("MODEL B: RANDOM FOREST - MULTI-CLASS MESSAGE TYPE")
print("="*60)

# Train Random Forest with hyperparameter tuning
print("\nTraining Random Forest with GridSearch...")

# Define parameter grid
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [20, 30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

# Initialize Random Forest
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# GridSearchCV
grid_search_rf = GridSearchCV(
    rf_base, param_grid_rf, cv=3, scoring='f1_weighted',
    verbose=1, n_jobs=-1
)

grid_search_rf.fit(X_train_multi, y_train_multi)

# Best model
rf_model = grid_search_rf.best_estimator_
print(f"\nBest parameters: {grid_search_rf.best_params_}")
print(f"Best cross-validation score: {grid_search_rf.best_score_:.4f}")

# Predictions
y_pred_rf = rf_model.predict(X_test_multi)
y_pred_rf_proba = rf_model.predict_proba(X_test_multi)

# Evaluation
print("\n" + "="*60)
print("MODEL B EVALUATION")
print("="*60)

print("\nConfusion Matrix:")
cm_rf = confusion_matrix(y_test_multi, y_pred_rf)
print(cm_rf)

print("\nClassification Report:")
print(classification_report(y_test_multi, y_pred_rf,
                          target_names=label_encoder.classes_, digits=4))

# Metrics
accuracy_rf = accuracy_score(y_test_multi, y_pred_rf)
precision_rf = precision_score(y_test_multi, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test_multi, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test_multi, y_pred_rf, average='weighted')

print(f"\nAccuracy:  {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall:    {recall_rf:.4f}")
print(f"F1-Score:  {f1_rf:.4f}")

# Feature importance
print("\nTop 20 Most Important Features:")

feature_names = (
    tfidf_multi.get_feature_names_out().tolist() +
    ['length', 'word_count', 'url_count', 'caps_ratio',
     'exclamation_count', 'question_count', 'digit_ratio',
     'special_char_count', 'phishing_kw_count', 
     'promotion_kw_count', 'scam_kw_count']
)

# Get feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:20]

valid_indices = [idx for idx in indices[:20] if idx < len(feature_names)]
for i, idx in enumerate(valid_indices):
    print(f"  {i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")

# Save model
with open('model_b_random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('tfidf_multi.pkl', 'wb') as f:
    pickle.dump(tfidf_multi, f)
with open('scaler_multi.pkl', 'wb') as f:
    pickle.dump(scaler_multi, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\nModel B saved successfully!")
print("="*60)
print("\n" + "="*60)
print("MODEL C: XGBOOST - MALWARE CLASSIFICATION")
print("="*60)

# Train XGBoost
print("\nTraining XGBoost Classifier...")

# Calculate scale_pos_weight for imbalanced data
unique, counts = np.unique(y_train_malware, return_counts=True)
if len(unique) == 2:
    scale_pos_weight = counts[0] / counts[1]
else:
    scale_pos_weight = 1

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb_model.fit(X_train_malware, y_train_malware)

# Predictions
y_pred_xgb = xgb_model.predict(X_test_malware)
y_pred_xgb_proba = xgb_model.predict_proba(X_test_malware)

# Evaluation
print("\n" + "="*60)
print("MODEL C EVALUATION")
print("="*60)

print("\nConfusion Matrix:")
cm_xgb = confusion_matrix(y_test_malware, y_pred_xgb)
print(cm_xgb)

# Get class names
if 'label_encoder_malware' in locals():
    target_names_malware = label_encoder_malware.classes_
else:
    target_names_malware = [f"Class_{i}" for i in unique]

print("\nClassification Report:")
print(classification_report(y_test_malware, y_pred_xgb,
                          target_names=target_names_malware, digits=4))

# Metrics
accuracy_xgb = accuracy_score(y_test_malware, y_pred_xgb)
precision_xgb = precision_score(y_test_malware, y_pred_xgb, average='weighted')
recall_xgb = recall_score(y_test_malware, y_pred_xgb, average='weighted')
f1_xgb = f1_score(y_test_malware, y_pred_xgb, average='weighted')

print(f"\nAccuracy:  {accuracy_xgb:.4f}")
print(f"Precision: {precision_xgb:.4f}")
print(f"Recall:    {recall_xgb:.4f}")
print(f"F1-Score:  {f1_xgb:.4f}")

# Feature importance
print("\nTop 20 Most Important Features:")
importances_xgb = xgb_model.feature_importances_
indices_xgb = np.argsort(importances_xgb)[::-1][:20]

for i, idx in enumerate(indices_xgb):
    print(f"{i+1}. {feature_cols[idx]}: {importances_xgb[idx]:.4f}")

# Cross-validation
print("\nCross-Validation Scores (3-fold):")
cv_scores_xgb = cross_val_score(xgb_model, X_train_malware, y_train_malware,
                                 cv=3, scoring='f1_weighted', n_jobs=-1)
print(f"F1 Scores: {cv_scores_xgb}")
print(f"Mean F1: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")

# Save model
with open('model_c_xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
with open('scaler_malware.pkl', 'wb') as f:
    pickle.dump(scaler_malware, f)

print("\nModel C saved successfully!")
print("="*60)
print("\n" + "="*60)
print("MODEL D: DBSCAN - SPAM PATTERN CLUSTERING")
print("="*60)

# Reduce dimensionality for DBSCAN (use sample for computational efficiency)
from sklearn.decomposition import TruncatedSVD

print("\nReducing dimensionality with TruncatedSVD...")
svd = TruncatedSVD(n_components=50, random_state=42)
X_clustering_reduced = svd.fit_transform(X_clustering)

print(f"Reduced shape: {X_clustering_reduced.shape}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

# Sample for DBSCAN (use stratified sampling if dataset is too large)
sample_size = min(10000, X_clustering_reduced.shape[0])
if X_clustering_reduced.shape[0] > sample_size:
    indices = np.random.choice(X_clustering_reduced.shape[0], 
                               sample_size, replace=False)
    X_dbscan = X_clustering_reduced[indices]
    labels_true = all_text_df['category'].values[indices]
else:
    X_dbscan = X_clustering_reduced
    labels_true = all_text_df['category'].values

print(f"\nClustering dataset size: {X_dbscan.shape[0]}")

# Train DBSCAN
print("\nTraining DBSCAN...")
dbscan_model = DBSCAN(eps=1.5, min_samples=5, n_jobs=-1)
clusters = dbscan_model.fit_predict(X_dbscan)

# Evaluation
print("\n" + "="*60)
print("MODEL D EVALUATION")
print("="*60)

n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print(f"\nNumber of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Percentage of noise: {100 * n_noise / len(clusters):.2f}%")

print("\nCluster distribution:")
cluster_counts = Counter(clusters)
for cluster_id, count in sorted(cluster_counts.items()):
    if cluster_id == -1:
        print(f"  Noise: {count}")
    else:
        print(f"  Cluster {cluster_id}: {count}")

# Silhouette score (excluding noise)
if n_clusters > 1:
    mask = clusters != -1
    if mask.sum() > 0:
        silhouette = silhouette_score(X_dbscan[mask], clusters[mask])
        print(f"\nSilhouette Score: {silhouette:.4f}")

# Analyze clusters
print("\nCluster Analysis (Dominant Category per Cluster):")
results_df = pd.DataFrame({
    'cluster': clusters,
    'true_category': labels_true
})

for cluster_id in sorted(set(clusters)):
    if cluster_id == -1:
        continue
    cluster_data = results_df[results_df['cluster'] == cluster_id]
    dominant_category = cluster_data['true_category'].mode()
    if len(dominant_category) > 0:
        dominant = dominant_category[0]
        percentage = (cluster_data['true_category'] == dominant).sum() / len(cluster_data) * 100
        print(f"  Cluster {cluster_id}: {dominant} ({percentage:.1f}%)")

# Save model
with open('model_d_dbscan.pkl', 'wb') as f:
    pickle.dump(dbscan_model, f)
with open('svd_clustering.pkl', 'wb') as f:
    pickle.dump(svd, f)

print("\nModel D saved successfully!")
print("="*60)
print("\n" + "="*60)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*60)

# Create comparison dataframe
comparison_data = {
    'Model': [
        'A: Naive Bayes',
        'B: Random Forest',
        'C: XGBoost',
        'D: DBSCAN'
    ],
    'Task': [
        'Binary Spam',
        'Multi-class Type',
        'Malware Classification',
        'Pattern Clustering'
    ],
    'Accuracy': [
        f"{accuracy_nb:.4f}",
        f"{accuracy_rf:.4f}",
        f"{accuracy_xgb:.4f}",
        'N/A (Unsupervised)'
    ],
    'Precision': [
        f"{precision_nb:.4f}",
        f"{precision_rf:.4f}",
        f"{precision_xgb:.4f}",
        'N/A'
    ],
    'Recall': [
        f"{recall_nb:.4f}",
        f"{recall_rf:.4f}",
        f"{recall_xgb:.4f}",
        'N/A'
    ],
    'F1-Score': [
        f"{f1_nb:.4f}",
        f"{f1_rf:.4f}",
        f"{f1_xgb:.4f}",
        'N/A'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n")
print(comparison_df.to_string(index=False))

print("\n" + "="*60)
# Create comprehensive evaluation visualizations
fig = plt.figure(figsize=(20, 12))

# 1. Confusion Matrix - Naive Bayes
plt.subplot(3, 4, 1)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Model A: Naive Bayes\nConfusion Matrix', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. ROC Curve - Naive Bayes
plt.subplot(3, 4, 2)
fpr_nb, tpr_nb, _ = roc_curve(y_test_binary, y_pred_nb_proba[:, 1])
plt.plot(fpr_nb, tpr_nb, label=f'ROC (AUC = {roc_auc_nb:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model A: ROC Curve', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 3. Metrics Comparison - Naive Bayes
plt.subplot(3, 4, 3)
metrics_nb = [accuracy_nb, precision_nb, recall_nb, f1_nb]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
bars = plt.bar(metric_names, metrics_nb, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
plt.ylim(0, 1.0)
plt.title('Model A: Performance Metrics', fontweight='bold')
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
plt.grid(axis='y', alpha=0.3)

# 4. Confusion Matrix - Random Forest
plt.subplot(3, 4, 5)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Model B: Random Forest\nConfusion Matrix', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)

# 5. Feature Importance - Random Forest
plt.subplot(3, 4, 6)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]

# Get feature names safely
manual_feature_cols = ['length', 'word_count', 'url_count', 'caps_ratio',
                       'exclamation_count', 'question_count', 'digit_ratio',
                       'special_char_count', 'phishing_kw_count', 
                       'promotion_kw_count', 'scam_kw_count']
feature_names = list(tfidf_multi.get_feature_names_out()) + manual_feature_cols

# Safety check: only use indices within bounds
top_indices = [i for i in indices if i < len(feature_names)][:15]
top_features = [feature_names[i] for i in top_indices]
top_importances = importances[top_indices]

plt.barh(range(len(top_features)), top_importances, color='green', alpha=0.7)
plt.yticks(range(len(top_features)), top_features, fontsize=7)
plt.xlabel('Importance')
plt.title('Model B: Top 15 Features', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# 6. Metrics Comparison - Random Forest
plt.subplot(3, 4, 7)
metrics_rf = [accuracy_rf, precision_rf, recall_rf, f1_rf]
bars = plt.bar(metric_names, metrics_rf, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
plt.ylim(0, 1.0)
plt.title('Model B: Performance Metrics', fontweight='bold')
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
plt.grid(axis='y', alpha=0.3)

# 7. Confusion Matrix - XGBoost
plt.subplot(3, 4, 9)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges',
            xticklabels=target_names_malware,
            yticklabels=target_names_malware)
plt.title('Model C: XGBoost\nConfusion Matrix', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)

# 8. Feature Importance - XGBoost
plt.subplot(3, 4, 10)
top_xgb = 15
top_indices_xgb = indices_xgb[:top_xgb]
top_features_xgb = [feature_cols[i] for i in top_indices_xgb]
top_importances_xgb = importances_xgb[top_indices_xgb]
plt.barh(range(top_xgb), top_importances_xgb, color='orange', alpha=0.7)
plt.yticks(range(top_xgb), top_features_xgb, fontsize=7)
plt.xlabel('Importance')
plt.title('Model C: Top 15 Features', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# 9. Metrics Comparison - XGBoost
plt.subplot(3, 4, 11)
metrics_xgb = [accuracy_xgb, precision_xgb, recall_xgb, f1_xgb]
bars = plt.bar(metric_names, metrics_xgb, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
plt.ylim(0, 1.0)
plt.title('Model C: Performance Metrics', fontweight='bold')
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
plt.grid(axis='y', alpha=0.3)

# 10. DBSCAN Cluster Distribution
plt.subplot(3, 4, 4)
cluster_sizes = [count for cluster_id, count in sorted(cluster_counts.items()) 
                 if cluster_id != -1]
cluster_labels = [f'C{cluster_id}' for cluster_id in sorted(set(clusters)) 
                  if cluster_id != -1]
if n_noise > 0:
    cluster_sizes.append(n_noise)
    cluster_labels.append('Noise')
plt.pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%', startangle=90)
plt.title('Model D: Cluster Distribution', fontweight='bold')

# 11. DBSCAN - Cluster Sizes
plt.subplot(3, 4, 8)
bars = plt.bar(cluster_labels, cluster_sizes, color='purple', alpha=0.7)
plt.xlabel('Cluster')
plt.ylabel('Number of Points')
plt.title('Model D: Points per Cluster', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# 12. Overall Model Comparison
plt.subplot(3, 4, 12)
models = ['NB', 'RF', 'XGB']
accuracies = [accuracy_nb, accuracy_rf, accuracy_xgb]
precisions = [precision_nb, precision_rf, precision_xgb]
recalls = [recall_nb, recall_rf, recall_xgb]
f1_scores = [f1_nb, f1_rf, f1_xgb]

x = np.arange(len(models))
width = 0.2

plt.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#3498db')
plt.bar(x - 0.5*width, precisions, width, label='Precision', color='#2ecc71')
plt.bar(x + 0.5*width, recalls, width, label='Recall', color='#f39c12')
plt.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#e74c3c')

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Classification Models Comparison', fontweight='bold')
plt.xticks(x, models)
plt.legend(fontsize=8)
plt.ylim(0, 1.0)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nModel evaluation visualization saved as 'model_evaluation_comparison.png'")
print("\n" + "="*60)
print("SAVING PROCESSED DATASETS")
print("="*60)

# Save cleaned and processed datasets
all_text_df.to_csv('processed_text_data.csv', index=False)
print("Saved: processed_text_data.csv")

malware_scaled_df.to_csv('processed_malware_data.csv', index=False)
print("Saved: processed_malware_data.csv")

# Save binary spam dataset
binary_spam_df.to_csv('binary_spam_dataset.csv', index=False)
print("Saved: binary_spam_dataset.csv")

# Save multiclass dataset with labels
multiclass_export = multiclass_df.copy()
multiclass_export['category_encoded'] = y_multiclass_encoded
multiclass_export.to_csv('multiclass_message_dataset.csv', index=False)
print("Saved: multiclass_message_dataset.csv")

print("\nAll processed datasets saved successfully!")
print("="*60)
print("\n" + "="*60)
print("MODEL PREDICTION EXAMPLES")
print("="*60)

# Example predictions for each model

# ========================================
# Model A: Binary Spam Detection
# ========================================
print("\n1. MODEL A - BINARY SPAM DETECTION")
print("-" * 60)

test_messages = [
    "Hi, how are you doing today?",
    "CONGRATULATIONS! You've WON a FREE iPhone! Click here NOW!!!",
    "Meeting scheduled for 2pm tomorrow in conference room B",
    "URGENT: Your bank account has been suspended. Verify immediately at http://fake-bank.com"
]

for msg in test_messages:
    # Clean text
    cleaned = clean_text(msg)
    # Vectorize
    vec = tfidf_binary.transform([cleaned])
    # Predict
    pred = nb_model.predict(vec)[0]
    proba = nb_model.predict_proba(vec)[0]
    
    print(f"\nMessage: {msg[:70]}...")
    print(f"Prediction: {'SPAM' if pred == 1 else 'HAM'}")
    print(f"Confidence: {proba[pred]:.2%}")

# ========================================
# Model B: Multi-class Message Type
# ========================================
print("\n\n2. MODEL B - MULTI-CLASS MESSAGE TYPE")
print("-" * 60)

test_messages_multi = [
    "Verify your account immediately or it will be suspended!",
    "50% OFF SALE! Limited time offer! Buy now and save!",
    "You have inherited $10 million from a Nigerian prince. Contact our lawyer.",
    "Hi John, the project report is attached. Let me know if you need anything."
]

for msg in test_messages_multi:
    # Clean and extract features
    cleaned = clean_text(msg)
    features = extract_text_features(msg)
    
    # TF-IDF
    vec_tfidf = tfidf_multi.transform([cleaned])
    
    # Manual features
    manual = np.array([[
        features['length'], features['word_count'], features['url_count'],
        features['caps_ratio'], features['exclamation_count'], 
        features['question_count'], features['digit_ratio'],
        features['special_char_count'], features['phishing_kw_count'],
        features['promotion_kw_count'], features['scam_kw_count']
    ]])
    manual_scaled = scaler_multi.transform(manual)
    
    # Combine
    from scipy.sparse import hstack
    vec_combined = hstack([vec_tfidf, manual_scaled])
    
    # Predict
    pred = rf_model.predict(vec_combined)[0]
    proba = rf_model.predict_proba(vec_combined)[0]
    category = label_encoder.inverse_transform([pred])[0]
    
    print(f"\nMessage: {msg[:70]}...")
    print(f"Prediction: {category}")
    print(f"Confidence: {proba[pred]:.2%}")
    print(f"Phishing keywords: {features['phishing_kw_count']}, "
          f"Promotion keywords: {features['promotion_kw_count']}, "
          f"Scam keywords: {features['scam_kw_count']}")

# ========================================
# Model C: Malware Detection
# ========================================
print("\n\n3. MODEL C - MALWARE DETECTION")
print("-" * 60)

# Take a few test samples from the test set
sample_indices = np.random.choice(len(X_test_malware), 5, replace=False)

for idx in sample_indices:
    features = X_test_malware[idx].reshape(1, -1)
    true_label = y_test_malware[idx]
    
    # Predict
    pred = xgb_model.predict(features)[0]
    proba = xgb_model.predict_proba(features)[0]
    
    if 'label_encoder_malware' in locals():
        true_class = label_encoder_malware.inverse_transform([true_label])[0]
        pred_class = label_encoder_malware.inverse_transform([pred])[0]
    else:
        true_class = f"Class_{true_label}"
        pred_class = f"Class_{pred}"
    
    print(f"\nSample {idx}:")
    print(f"True Label: {true_class}")
    print(f"Predicted: {pred_class}")
    print(f"Confidence: {proba[pred]:.2%}")
    print(f"Match: {'✓' if pred == true_label else '✗'}")

print("\n" + "="*60)
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)

summary = f"""
SPAM/MALWARE DETECTION ML PROJECT
==================================

DATASETS PROCESSED:
- Email Spam: {len(emails_df)} messages
- SMS Spam: {len(sms_df)} messages  
- Phishing Emails: {len(phishing_df)} messages
- Malware Processes: {len(malware_df)} samples
- Total Text Messages: {len(all_text_df)}

MODELS TRAINED:
---------------

1. MODEL A - Naive Bayes (Binary Spam Detection)
   - Algorithm: MultinomialNB
   - Dataset: Emails + SMS
   - Accuracy: {accuracy_nb:.4f}
   - Precision: {precision_nb:.4f}
   - Recall: {recall_nb:.4f}
   - F1-Score: {f1_nb:.4f}
   - ROC-AUC: {roc_auc_nb:.4f}

2. MODEL B - Random Forest (Multi-class Message Type)
   - Algorithm: Random Forest Classifier
   - Dataset: All text messages
   - Classes: {', '.join(label_encoder.classes_)}
   - Accuracy: {accuracy_rf:.4f}
   - Precision: {precision_rf:.4f}
   - Recall: {recall_rf:.4f}
   - F1-Score: {f1_rf:.4f}

3. MODEL C - XGBoost (Malware Classification)
   - Algorithm: XGBoost Classifier
   - Dataset: Process features (100K samples)
   - Features: {len(feature_cols)} numerical features
   - Accuracy: {accuracy_xgb:.4f}
   - Precision: {precision_xgb:.4f}
   - Recall: {recall_xgb:.4f}
   - F1-Score: {f1_xgb:.4f}

4. MODEL D - DBSCAN (Pattern Clustering)
   - Algorithm: DBSCAN
   - Dataset: All text messages
   - Clusters Found: {n_clusters}
   - Noise Points: {n_noise} ({100*n_noise/len(clusters):.1f}%)

KEY INSIGHTS:
-------------
1. Spam messages are characterized by:
   - Higher CAPS ratio
   - More exclamation marks
   - Presence of promotional/phishing keywords
   - URLs and suspicious links

2. Message type classification successfully identifies:
   - Phishing attempts (credential harvesting)
   - Promotional spam (marketing/sales)
   - Scam messages (financial fraud)
   - Safe communications

3. Malware classification leverages:
   - Process memory metrics
   - CPU time patterns
   - System state indicators

4. Clustering revealed distinct spam campaign patterns

FILES GENERATED:
----------------
- Model files: model_a_naive_bayes.pkl, model_b_random_forest.pkl,
              model_c_xgboost.pkl, model_d_dbscan.pkl
- Preprocessors: tfidf_binary.pkl, tfidf_multi.pkl, scaler_*.pkl, label_encoder.pkl
- Datasets: processed_text_data.csv, processed_malware_data.csv
- Visualizations: text_data_analysis.png, malware_data_analysis.png,
                 model_evaluation_comparison.png

CYBERSECURITY APPLICATIONS:
---------------------------
✓ Email/SMS spam filtering for end users
✓ Phishing detection for enterprise security
✓ Malware process identification for system protection
✓ Threat intelligence through pattern clustering
"""

print(summary)

# Save summary to file
with open('project_summary.txt', 'w') as f:
    f.write(summary)

print("\nProject summary saved to 'project_summary.txt'")
print("="*60)