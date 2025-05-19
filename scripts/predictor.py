import joblib
from offers import recommend_mtn_offer
import pandas as pd

def predict_user_bundle(df_input):
    # 1. Preprocess: clean, format dates, etc.
    df_clean = preprocess_user_data(df_input)

    # 2. Feature engineer: group by user to get avg_data_before, std, recharge_before, etc.
    df_features = generate_features(df_clean)

    # 3. Load saved model
    model = joblib.load("models/final_xgb.pkl")

    feature_cols = ['avg_data_before_upgrade', 'std_before', 'total_recharge_before', 'increased_usage']

    # 4. Predict increased usage
    df_features['increased_usage'] = model.predict(df_features[feature_cols])

    # 5. Segment the user
    df_features['user_segment'] = df_features.apply(segment_user_row, axis=1)

    # 6. Recommend campaign
    df_features['recommended_campaign'] = df_features.apply(recommend_mtn_offer, axis=1)

    return df_features[['user', 'user_segment', 'increased_usage', 'recommended_campaign']]

def preprocess_user_data(df):
    #let's first impute nan values in total_reload_ammount with the mean of those who reloaded once
    mean_total = df[(df['total_reloads'] == 1) & (df['total_reload_amount'].notna())]['total_reload_amount'].mean()
    
    df['total_reload_amount'] = df['total_reload_amount'].fillna(mean_total)

    #let's replace the character '-' with 6 if found and convert the the dtype to int

    df['device_category'].value_counts()
    df['device_category'] = df['device_category'].replace('-','6')
    df['device_category'] = df['device_category'].astype(int)

    return df

def generate_features(df):
    # convert dates columns to datetime
    df['dates'] = pd.to_datetime(df['dates'], format='%Y%m%d')

    # Sort the DataFrame by 'Phone Number' and 'dates'
    df = df.sort_values(by=['Phone Number', 'dates'])

    # Identify smartphones and those that aren't
    df['isSmartphone'] = df['device_category'] == 5

    #For each user, find the first date they used a smartphone

    dates_upgraded = df[df['isSmartphone']].groupby('Phone Number')['dates'].min().reset_index()
    dates_upgraded = dates_upgraded.rename(columns={'dates': 'dates_upgraded', 'Phone Number': 'user'})

    # Let's get users who used feature/basic phone before

    feature_or_basic = df[df['device_category'].isin([4,7])]['Phone Number'].unique()

    dates_upgraded =  dates_upgraded[dates_upgraded['user'].isin(feature_or_basic)]
    
    # Merge date_upgraded with the original df

    #Firstly, we need to rename the 'Phone Number' column to 'user' in the original DataFrame
    df = df.rename(columns={'Phone Number': 'user'})

    df = df.merge(dates_upgraded, on='user', how='left')

    # Create columns to label rows as "before" or "after" upgrade

    df['Days_to_Upgrade'] = (df['dates']- df['dates_upgraded']).dt.days

    df['isBefore'] = df['Days_to_Upgrade'].between(-30, -1) #30 days before upgrade
    df['isAfter'] = df['Days_to_Upgrade'].between(0, 60) #60 days after upgrade

    before = df[df['isBefore']].groupby('user').agg(
        avg_data_before_upgrade=('data_kb', 'mean'),
        std_before=('data_kb', 'std'),
        total_recharge_before=('total_reload_amount', 'sum')
    ).reset_index()

    after = df[df['isAfter']].groupby('user').agg(
        avg_data_after_upgrade=('data_kb', 'mean'),
        std_after=('data_kb', 'std'),
        total_recharge_after=('total_reload_amount', 'sum'),
        days_active_after=('data_kb', lambda x: (x > 0).sum()),
        time_to_first_data_use=('Days_to_Upgrade', lambda x: x[df.loc[x.index, 'data_kb'] > 0].min())
    ).reset_index()

    # Merge them together

    features = before.merge(after, on='user', how='outer')

    features['recharge_growth'] = features['total_recharge_after'] - features['total_recharge_before']

    features['increased_usage'] = (
        (features['avg_data_before_upgrade'].notna()) &
        (features['avg_data_before_upgrade'] > 0) &
        (features['avg_data_after_upgrade'] > features['avg_data_before_upgrade']*1.5)
    ).astype(int)

    features['data_flag'] = features['avg_data_before_upgrade'].isna().astype(int)
    features['std_flag'] = features['std_before'].isna().astype(int)
    features['recharge_flag'] = features['total_recharge_before'].isna().astype(int)

    # Fill missing values with 0

    features['avg_data_before_upgrade'] = features['avg_data_before_upgrade'].fillna(0)
    features['std_before'] = features['std_before'].fillna(0)
    features['total_recharge_before'] = features['total_recharge_before'].fillna(0)
    
    return features



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