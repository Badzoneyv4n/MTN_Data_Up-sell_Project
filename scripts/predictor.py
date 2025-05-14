import joblib
from recommender import recommend_mtn_offer

def predict_user_bundle(df_input):
    # 1. Preprocess: clean, format dates, etc.
    df_clean = preprocess_user_data(df_input)

    # 2. Feature engineer: group by user to get avg_data_before, std, recharge_before, etc.
    df_features = generate_features(df_clean)

    # 3. Load saved model
    model = joblib.load("models/final_xgb.pkl")

    feature_cols = []

    # 4. Predict increased usage
    df_features['increased_usage'] = model.predict(df_features[feature_cols])

    # 5. Segment the user
    df_features['user_segment'] = df_features.apply(segment_user_row, axis=1)

    # 6. Recommend campaign
    df_features['recommended_campaign'] = df_features.apply(recommend_mtn_offer, axis=1)

    return df_features[['user', 'user_segment', 'increased_usage', 'recommended_campaign']]

def preprocess_user_data(df):
    pass

def generate_features(df):
    pass

def segment_user_row(row):
    low_threshold = row['avg_data_after_upgrade'].quantile(0.25)
    high_threshold = row['avg_data_after_upgrade'].quantile(0.75)
    
    usage = row['avg_data_after_upgrade']
    
    if usage == 0:
        return 'non'
    elif usage < low_threshold:
        return 'low'
    elif usage < high_threshold:
        return 'medium'
    else:
        return 'high'