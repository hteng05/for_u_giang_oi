"""
Simple Prediction Script for Spam/Malware Detection Models
Edit the test inputs below and run: python predict.py
"""

import pickle
import numpy as np
import pandas as pd
import re
from scipy.sparse import hstack

# ============================================================================
# LOAD ALL MODELS AND PREPROCESSORS
# ============================================================================

print("Loading models and preprocessors...")

# Model A - Binary Spam
with open('model_a_naive_bayes.pkl', 'rb') as f:
    nb_model = pickle.load(f)
with open('tfidf_binary.pkl', 'rb') as f:
    tfidf_binary = pickle.load(f)

# Model B - Multi-class
with open('model_b_random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('tfidf_multi.pkl', 'rb') as f:
    tfidf_multi = pickle.load(f)
with open('scaler_multi.pkl', 'rb') as f:
    scaler_multi = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Model C - Malware
with open('model_c_xgboost.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('scaler_malware.pkl', 'rb') as f:
    scaler_malware = pickle.load(f)

print("All models loaded successfully!\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_features(text):
    """Extract all text features for classification"""
    if not isinstance(text, str) or len(text) == 0:
        return {
            'length': 0, 'word_count': 0, 'url_count': 0, 'email_count': 0,
            'caps_ratio': 0, 'exclamation_count': 0, 'question_count': 0,
            'digit_ratio': 0, 'special_char_count': 0,
            'phishing_kw_count': 0, 'promotion_kw_count': 0, 'scam_kw_count': 0
        }
    
    text_lower = text.lower()
    length = len(text)
    word_count = len(text.split())
    url_count = len(re.findall(r'http|www', text_lower))
    email_count = len(re.findall(r'\S+@\S+', text_lower))
    caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
    caps_ratio = caps_words / word_count if word_count > 0 else 0
    exclamation_count = text.count('!')
    question_count = text.count('?')
    digit_count = sum(c.isdigit() for c in text)
    digit_ratio = digit_count / length if length > 0 else 0
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    
    # Keywords
    phishing_keywords = ['verify', 'account', 'suspended', 'login', 'password',
                        'confirm', 'security', 'urgent', 'click here', 'update',
                        'bank', 'paypal', 'credential', 'expire', 'validation']
    promotion_keywords = ['free', 'win', 'winner', 'prize', 'offer', 'discount',
                         'sale', 'limited time', 'buy now', 'deal', 'save',
                         'congratulations', 'claim', 'bonus', 'gift', 'reward']
    scam_keywords = ['lottery', 'inheritance', 'transfer', 'western union',
                    'nigeria', 'prince', 'million', 'beneficiary', 'fund',
                    'lawyer', 'diplomat', 'courier', 'tax refund', 'government grant']
    
    phishing_kw = sum(1 for kw in phishing_keywords if kw in text_lower)
    promotion_kw = sum(1 for kw in promotion_keywords if kw in text_lower)
    scam_kw = sum(1 for kw in scam_keywords if kw in text_lower)
    
    return {
        'length': length, 'word_count': word_count, 'url_count': url_count,
        'email_count': email_count, 'caps_ratio': caps_ratio,
        'exclamation_count': exclamation_count, 'question_count': question_count,
        'digit_ratio': digit_ratio, 'special_char_count': special_chars,
        'phishing_kw_count': phishing_kw, 'promotion_kw_count': promotion_kw,
        'scam_kw_count': scam_kw
    }

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_binary_spam(message):
    """Model A: Predict if message is spam or ham"""
    cleaned = clean_text(message)
    vec = tfidf_binary.transform([cleaned])
    pred = nb_model.predict(vec)[0]
    proba = nb_model.predict_proba(vec)[0]
    
    result = "SPAM" if pred == 1 else "HAM"
    confidence = proba[pred] * 100
    
    return result, confidence

def predict_message_type(message):
    """Model B: Predict message type (Safe, Phishing, Promotion, Scam)"""
    cleaned = clean_text(message)
    features = extract_text_features(message)
    
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
    vec_combined = hstack([vec_tfidf, manual_scaled])
    
    pred = rf_model.predict(vec_combined)[0]
    proba = rf_model.predict_proba(vec_combined)[0]
    category = label_encoder.inverse_transform([pred])[0]
    confidence = proba[pred] * 100
    
    return category, confidence, features

def predict_malware(process_features):
    """Model C: Predict if process is malware"""
    if len(process_features) == 34:
        process_features = process_features[:32]
    
    # Define column names (must match training data)
    feature_names = ['state', 'usage_counter', 'prio', 'static_prio', 'normal_prio',
                     'policy', 'vm_pgoff', 'vm_truncate_count', 'task_size',
                     'cached_hole_size', 'free_area_cache', 'mm_users', 'map_count',
                     'hiwater_rss', 'total_vm', 'shared_vm', 'exec_vm', 'reserved_vm',
                     'nr_ptes', 'end_data', 'last_interval', 'nvcsw', 'nivcsw',
                     'min_flt', 'maj_flt', 'fs_excl_counter', 'lock', 'utime',
                     'stime', 'gtime', 'cgtime', 'signal_nvcsw']
    
    # Create DataFrame instead of array
    features_df = pd.DataFrame([process_features], columns=feature_names)
    
    features_scaled = scaler_malware.transform(features_df)
    pred = xgb_model.predict(features_scaled)[0]
    proba = xgb_model.predict_proba(features_scaled)[0]
    confidence = proba[pred] * 100
    return int(pred), confidence

# ============================================================================
# TEST INPUTS - EDIT THESE TO TEST YOUR OWN DATA
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("SPAM/MALWARE DETECTION - PREDICTION DEMO")
    print("="*80)
    
    # ========================================================================
    # TEST 1: BINARY SPAM DETECTION
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: BINARY SPAM DETECTION (Model A)")
    print("="*80)
    
    test_messages = [
        "I love your outfit today baby nepo",
        "CONGRATULATIONS! You've WON a FREE iPhone 15! Click here NOW to claim!!!",
        "Your package will arrive tomorrow between 2-4pm.",
        "URGENT: Your bank account suspended. Verify immediately at http://fake-bank.com"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        result, conf = predict_binary_spam(msg)
        print(f"\n{i}. Message: {msg[:70]}...")
        print(f"   Prediction: {result}")
        print(f"   Confidence: {conf:.1f}%")
    
    # ========================================================================
    # TEST 2: MULTI-CLASS MESSAGE TYPE
    # ========================================================================
    print("\n\n" + "="*80)
    print("TEST 2: MESSAGE TYPE CLASSIFICATION (Model B)")
    print("="*80)
    
    test_messages_multi = [
        "I love Giang's cooking! Let's have dinner together.",
        "Baby come to my room",
        "50% OFF SALE! Limited time offer! Buy now and save big!!!",
        "You have inherited $10 million from Nigerian prince. Contact lawyer now."
    ]
    
    for i, msg in enumerate(test_messages_multi, 1):
        category, conf, features = predict_message_type(msg)
        print(f"\n{i}. Message: {msg[:70]}...")
        print(f"   Prediction: {category}")
        print(f"   Confidence: {conf:.1f}%")
        print(f"   Keywords - Phishing: {features['phishing_kw_count']}, "
              f"Promotion: {features['promotion_kw_count']}, "
              f"Scam: {features['scam_kw_count']}")
    
    # ========================================================================
    # TEST 3: MALWARE DETECTION
    # ========================================================================
    print("\n\n" + "="*80)
    print("TEST 3: MALWARE DETECTION (Model C)")
    print("="*80)
    
    # Example: 34 process features (modify these values)
    # These are random examples - replace with real process metrics
    test_process_1 = [0,0,3069378560,14274,0,0,0,13173,0,0,24,724,6850,0,150,120,124,210,0,120,3473,341974,0,0,120,0,3204448256,380690,4,0,0,0]  # 32 values

    test_process_2 = [1, 250, 1, 1, 1, 2, 2, 2, 1, 1, 1, 8192, 1, 1, 2, 10,
                    200, 1000, 20, 100, 5, 20, 16384, 2000, 10, 5, 20, 1, 1, 1,
                    500, 100]  # 32 values
    
    print("\nProcess 1:")
    pred, conf = predict_malware(test_process_1)
    print(f"   Prediction: {'MALWARE' if pred == 1 else 'BENIGN'} (Class {pred})")
    print(f"   Confidence: {conf:.1f}%")
    
    print("\nProcess 2:")
    pred, conf = predict_malware(test_process_2)
    print(f"   Prediction: {'MALWARE' if pred == 1 else 'BENIGN'} (Class {pred})")
    print(f"   Confidence: {conf:.1f}%")
    
    # ========================================================================
    # CUSTOM INPUT SECTION - UNCOMMENT TO USE
    # ========================================================================
    print("\n\n" + "="*80)
    print("CUSTOM INPUT TEST")
    print("="*80)
    
    # Test your own message
    custom_message = "Hey, check out this amazing offer at http://best-deals.com!!!"
    
    print(f"\nYour message: {custom_message}")
    print("\nModel A (Binary):")
    result, conf = predict_binary_spam(custom_message)
    print(f"  {result} ({conf:.1f}% confidence)")
    
    print("\nModel B (Multi-class):")
    category, conf, features = predict_message_type(custom_message)
    print(f"  {category} ({conf:.1f}% confidence)")
    
    print("\n" + "="*80)
    print("PREDICTION DEMO COMPLETE")
    print("="*80)