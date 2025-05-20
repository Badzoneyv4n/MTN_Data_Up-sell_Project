import joblib
import pandas as pd, numpy as np
from .offers import recommend_mtn_offer

# Load pre-trained models
avg_model = joblib.load("../output/models/avg_data_xgb.pkl")
incr_usage_model = joblib.load("../output/models/increased_usage_xgb.pkl")
rech_grow_model = joblib.load("../output/models/recharge_growth_xgb.pkl")
user_seg_model = joblib.load("../output/models/user_segment_xgb.pkl")

def recommend(df_input: pd.DataFrame, mode='auto')-> pd.DataFrame:
    """
    Main recommendation system entry point.
    Handles all input types and returns DataFrame with predictions and recommendations.
    """
    try:
        df_output = pd.DataFrame()

        # Automatically detect the input mode if not specified
        if mode == 'auto':
            mode = detect_input_mode(df_input)

        if mode == 'single':
            print("Single user mode detected.")
            df_output = handle_single_user(df_input)

        elif mode == 'multi':
            print("Multiple users mode detected.")
            df_output = handle_multi_user(df_input)

        elif mode == 'direct':
            print("Direct input mode detected.")
            if not isinstance(df_input, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame for 'direct' mode.")
            df_output = handle_direct_input(df_input)
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'single', 'multi', 'direct', or 'auto'.")
        
        # Convert user_segment to string labels
        df_output['user_segment'] = df_output['user_segment'].astype(str)
        df_output['user_segment'] = df_output['user_segment'].replace({
            '0': 'non',
            '1': 'low',
            '2': 'medium',
            '3': 'high'
        })

        # Generate offer and reason for each user
        offers = df_output.apply(recommend_mtn_offer, axis=1)
        df_output['offer'] = offers.apply(lambda x: x['offer'])
        df_output['reason'] = offers.apply(lambda x: x['reason'])

        return df_output

    except Exception as e:
        raise RuntimeError(f'Error in generate_features: {e}')

def detect_input_mode(df):
    """
    Determines the input type: direct features or raw usage logs (single or multi user).
    """
    try:
        # If direct features are present, use direct mode
        if {'avg_data_before_upgrade', 'std_before', 'total_recharge_before'}.issubset(df.columns):
            return 'direct'
        # If only one phone number, treat as single user
        if df['Phone Number'].nunique() == 1:
            return 'single'
        # Otherwise, treat as multi-user
        return 'multi'
    except Exception as e:
        raise RuntimeError(f'Error in detect_input_mode: {e}')

def handle_single_user(df_input):
    """
    Handles a single user's raw usage logs.
    Cleans, generates features, and runs the prediction pipeline.
    """
    print('Handling single user')
    try: 
        df_cleaned = clean_and_process(df_input)
        df_new = generate_features(df_cleaned)
        # Run full prediction pipeline
        df_incr_usage = predict_increased_usage(df_new)
        df_user_segment = predict_user_segment(df_incr_usage)
        df_avg_data_after = predict_avg_data_after(df_user_segment)
        df_recharge_growth = predict_recharge_growth(df_avg_data_after)
        return df_recharge_growth
    except Exception as e:
        raise RuntimeError(f'Error in handle_single_user: {e}')

def handle_multi_user(df_input):
    """
    Handles multiple users' raw usage logs.
    Cleans, generates features, and runs the prediction pipeline.
    """
    print('Handling multiple users')
    try:
        df_cleaned = clean_and_process(df_input)
        df_new = generate_features(df_cleaned)
        # Run full prediction pipeline
        df_incr_usage = predict_increased_usage(df_new)
        df_user_segment = predict_user_segment(df_incr_usage)
        df_avg_data_after = predict_avg_data_after(df_user_segment)
        df_recharge_growth = predict_recharge_growth(df_avg_data_after)
        return df_recharge_growth
    except Exception as e:  
        raise RuntimeError(f'Error in handle_multi_user: {e}')

def handle_direct_input(df_input): 
    """
    Handles direct input of pre-computed features for one or more users.
    Runs the prediction pipeline from the feature stage.
    """
    print('Handling direct input')
    try:
        features = {
            'data_flag',
            'avg_data_before_upgrade',
            'std_flag',
            'std_before',
            'total_recharge_before',
            'recharge_flag'
        }
        # Check for missing features
        missing = set(features) - set(df_input.columns)
        if missing:
            raise ValueError(
                f"handle_direct_input: missing columns {missing}. "
                f"Expected: {features}, got: {set(df_input.columns)}"
            )
        # Prepare input
        X = df_input[list(features)].copy()
        X = X.apply(pd.to_numeric, errors='coerce')
        # Run full prediction pipeline
        df_incr_usage = predict_increased_usage(X)
        df_user_segment = predict_user_segment(df_incr_usage)
        df_avg_data_after = predict_avg_data_after(df_user_segment)
        df_recharge_growth = predict_recharge_growth(df_avg_data_after)
        return df_recharge_growth
    except Exception as e:
        raise RuntimeError(f'Error in handle_direct_input: {e}')

def predict_user_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts user segment and confidence.
    """
    features = [
        'avg_data_before_upgrade', 
        'std_before', 
        'total_recharge_before', 
        'increased_usage',
        'data_flag',
        'std_flag',
        'recharge_flag',
        'upgrade_prob_confidence'
    ]
    # Check for missing features
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(
            f"predict_user_segment: missing columns {missing}. "
            f"Expected: {features}, got: {set(df.columns)}"
        )
    # Prepare input
    X = df[list(features)].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    # Predict segment and probability
    y_pred = user_seg_model.predict(X)
    proba = user_seg_model.predict_proba(X)
    X['user_segment'] = y_pred
    X['user_segment_prob_confidence'] = proba[np.arange(len(y_pred)), y_pred]
    # Add user column back if present
    if 'user' in df.columns:
        X['user'] = df['user'].values
    return X

def predict_recharge_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts recharge growth for each user.
    """
    try:
        features = [
            'avg_data_after_upgrade',
            'avg_data_before_upgrade',
            'increased_usage',
            'std_before',
            'total_recharge_before',
            'user_segment'
        ]
        # Check for missing features
        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(
                f"predict_recharge_growth: missing columns {missing}. "
                f"Expected: {features}, got: {set(df.columns)}"
            )
        # Prepare input
        X = df[list(features)].copy()
        X = X.apply(pd.to_numeric, errors='coerce')
        # Predict recharge growth
        y_pred = rech_grow_model.predict(X)
        X['recharge_growth'] = y_pred
        # Add user column back if present
        if 'user' in df.columns:
            X['user'] = df['user'].values
        return X
    except Exception as e:
        raise RuntimeError(f'Error in predict_recharge_growth: {e}')

def predict_avg_data_after(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts average data usage after upgrade for each user.
    """
    try:
        features = [
            'avg_data_before_upgrade',
            'std_before',
            'total_recharge_before',
            'increased_usage',
            'data_flag',
            'std_flag',
            'recharge_flag',
            'upgrade_prob_confidence',
            'user_segment',
            'user_segment_prob_confidence'
        ]
        # Check for missing features
        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(
                f"predict_avg_data_after: missing columns {missing}. "
                f"Expected: {features}, got: {set(df.columns)}"
            )
        # Prepare input
        X = df[list(features)].copy()
        # Cast user_segment as category for model
        X['user_segment'] = X['user_segment'].astype('category')
        # Predict average data after upgrade
        y_pred = avg_model.predict(X)
        X['avg_data_after_upgrade'] = y_pred
        # Add user column back if present
        if 'user' in df.columns:
            X['user'] = df['user'].values
        return X
    except Exception as e:
        raise RuntimeError(f'Error in predict_avg_data_after: {e}')

def predict_increased_usage(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts increased usage and confidence for each user.
    """
    try:
        features = [
            'avg_data_before_upgrade',
            'std_before',
            'total_recharge_before',
            'data_flag',
            'std_flag',
            'recharge_flag'
        ]
        # Check for missing features
        missing = set(features) - set(input_df.columns)
        if missing:
            raise ValueError(
                f"predict_increased_usage: missing columns {missing}. "
                f"Expected: {features}, got: {set(input_df.columns)}"
            )
        # Prepare input
        X = input_df[list(features)].copy()
        X = X.apply(pd.to_numeric, errors='coerce')
        # Predict increased usage and probability
        y_pred = incr_usage_model.predict(X)
        y_pred_prob = incr_usage_model.predict_proba(X)[:, 1]
        X['increased_usage'] = y_pred 
        X['upgrade_prob_confidence'] = y_pred_prob
        # Add user column back if present
        if 'user' in input_df.columns:
            X['user'] = input_df['user'].values
        return X
    except Exception as e:
        raise RuntimeError(f'Error in predict_increased_usage: {e}')

def clean_and_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and processes raw usage logs for feature engineering.
    - Imputes missing values
    - Filters device categories
    - Converts columns to correct types
    - Renames 'Phone Number' to 'user'
    """
    try:
        features_expected = {
            'dates',
            'Phone Number',
            'total_reloads',
            'total_reload_amount',
            'device_category',
            'data_kb'
        }
        # Check for missing columns
        missing = features_expected - set(df.columns)
        if missing:
            raise ValueError(
                f"clean_and_process: missing columns {missing}. "
                f"Expected: {features_expected}, got: {set(df.columns)}"
            )
        # Impute missing total_reload_amount with mean
        if int(df['total_reload_amount'].isnull().sum()) > 0:
            m = df[(df['total_reloads'] == 1) & (df['total_reload_amount'].notna())]['total_reload_amount'].mean()
            df = df.copy()
            df['total_reload_amount'] = df['total_reload_amount'].fillna(m)
            print(f"Imputed missing values in total_reload_amount using mean: {m:.2f}")
        # Filter device categories and convert to int
        df = df[df['device_category'] != '-'].copy()
        df['device_category'] = df['device_category'].astype(int)
        dev_expected = {5, 7, 4}
        present_cats = set(df['device_category'].unique())
        if not present_cats & dev_expected:
            raise ValueError(f"No expected device categories found in device_category column. Found: {present_cats}")
        # Remove smartphones if present
        if 5 in present_cats:
            df = df[df['device_category'] != 5]
            print("Removed all rows where device_category == 5 (smartphone).")
        # Ensure at least one feature/basic phone row
        if not df['device_category'].isin([7, 4]).any():
            raise ValueError("No rows with device_category 7 or 4 (feature phones) found after filtering.")
        # Convert dates to datetime and sort
        df['dates'] = pd.to_datetime(df['dates'], format='%Y%m%d')
        df = df.sort_values(by=['Phone Number', 'dates'])
        # Rename 'Phone Number' to 'user'
        df = df.rename(columns={'Phone Number': 'user'})
        # Keep only feature/basic phone users
        df = df[df['device_category'].isin([4,7])]
        return df
    except Exception as e:
        raise RuntimeError(f'Error in clean_and_process: {e}')  

def generate_features(df: pd.DataFrame ) -> pd.DataFrame:
    """
    Aggregates and engineers features from cleaned logs.
    - Groups by user
    - Computes mean, std, sum
    - Adds missing value flags
    """
    try: 
        features_expected = {
            'dates',
            'user',
            'total_reloads',
            'total_reload_amount',
            'device_category',
            'data_kb'
        }
        # Check for missing columns
        missing = features_expected - set(df.columns)
        if missing:
            raise ValueError(
                f"generate_features: missing columns {missing}. "
                f"Expected: {features_expected}, got: {set(df.columns)}"
            )
        # Ensure 'user' column exists
        if 'user' not in df.columns:
            raise ValueError("generate_features: 'user' column missing after preprocessing.")
        # Check for empty DataFrame
        if df.empty:
            raise ValueError("generate_features: Input DataFrame is empty.")
        # Ensure correct types for calculations
        for col in ['total_reloads', 'total_reload_amount', 'device_category', 'data_kb']:
            if col in df.columns:
                if col == 'device_category':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        # Aggregate usage stats per user
        df = df.groupby('user').agg(
            avg_data_before_upgrade=('data_kb', 'mean'),
            std_before=('data_kb', 'std'),
            total_recharge_before=('total_reload_amount', 'sum')
        ).reset_index()
        # Handle Missing values
        df['data_flag'] = df['avg_data_before_upgrade'].isna().astype(int)
        df['std_flag'] = df['std_before'].isna().astype(int)
        df['recharge_flag'] = df['total_recharge_before'].isna().astype(int)
        # Fill missing values with 0
        df['avg_data_before_upgrade'] = df['avg_data_before_upgrade'].fillna(0)
        df['std_before'] = df['std_before'].fillna(0)
        df['total_recharge_before'] = df['total_recharge_before'].fillna(0)
        return df
    except Exception as e:
        raise RuntimeError(f'Error in generate_features: {e}')


# import joblib
# import pandas as pd , numpy as np
# from offers import recommend_mtn_offer

# avg_model = joblib.load("../output/models/avg_data_xgb.pkl")
# incr_usage_model = joblib.load("../output/models/increased_usage_xgb.pkl")
# rech_grow_model = joblib.load("../output/models/recharge_growth_xgb.pkl")
# user_seg_model = joblib.load("../output/models/user_segment_xgb.pkl")

# def recommend(df_input: pd.DataFrame, mode='auto')-> pd.DataFrame:
#     """
#     Main recommendation system entry point.

#     Parameters:
#     - df_input: pandas DataFrame of user data
#     - mode: 'auto', 'single', 'multi', or 'direct'

#     Returns:
#     - DataFrame with prediction, segment, recommendation, and reason
#     """

#     try:
#         df_output = pd.DataFrame()

#         # Automatically detect the input mode
#         if mode == 'auto':
#             mode = detect_input_mode(df_input)

#         if mode == 'single':
#             print("Single user mode detected.")
#             df_output = handle_single_user(df_input)

#         elif mode == 'multi':
#             print("Multiple users mode detected.")
#             df_output = handle_multi_user(df_input)

#         elif mode == 'direct':
#             print("Direct input mode detected.")
#             #Remember to handle input as a dataframe
#             if not isinstance(df_input, pd.DataFrame):
#                 raise ValueError("Input must be a pandas DataFrame for 'direct' mode.")
#             df_output = handle_direct_input(df_input)
#         else:
#             raise ValueError(f"Unsupported mode '{mode}'. Use 'single', 'multi', 'direct', or 'auto'.")
        
#         df_output['user_segment'] = df_output['user_segment'].astype(str)
    
#         df_output['user_segment'] = df_output['user_segment'].replace({
#             '0': 'non',
#             '1': 'low',
#             '2': 'medium',
#             '3': 'high'
#         })

#         offers = df_output.apply(recommend_mtn_offer, axis=1)
#         df_output['offer'] = offers.apply(lambda x: x['offer'])
#         df_output['reason'] = offers.apply(lambda x: x['reason'])

#         return df_output

#     except Exception as e:
#         raise RuntimeError(f'Error in generate_features: {e}')

# def detect_input_mode(df):
#     """
#     Determines the input type: direct features or raw usage logs (single or multi user).
#     """
#     try:
#         if {'avg_data_before_upgrade', 'std_before', 'total_recharge_before'}.issubset(df.columns):
#             return 'direct'
        
#         if df['Phone Number'].nunique() == 1:
#             return 'single'

#         return 'multi'

#     except Exception as e:
#         raise RuntimeError(f'Error in detect_input_mode: {e}')


# def handle_single_user(df_input):
#     """
#     We're expecting a dataframe wich contains informations of 1 user.
#     so we calculate everything on him
#     """
#     print('Handling single user')

#     try: 
#         # clean the dataframe
#         df_cleaned = clean_and_process(df_input)

#         # generate features
#         df_new = generate_features(df_cleaned)

#         # predict increased usage
#         df_incr_usage = predict_increased_usage(df_new)

#         # predict user segment
#         df_user_segment = predict_user_segment(df_incr_usage)

#         #predict avg data after upgrade
#         df_avg_data_after = predict_avg_data_after(df_user_segment)

#         #predict recharge growth
#         df_recharge_growth = predict_recharge_growth(df_avg_data_after)

#         return df_recharge_growth
    
#     except Exception as e:
#         raise RuntimeError(f'Error in handle_single_user: {e}')


# def handle_multi_user(df_input):
#     """
#     We're expecting a dataframe wich contains informations of multiple users.  
#     so we calculate everything on them
#     """
#     print('Handling multiple users')

#     try:
#         # clean the dataframe
#         df_cleaned = clean_and_process(df_input)

#         # generate features
#         df_new = generate_features(df_cleaned)

#         # predict increased usage
#         df_incr_usage = predict_increased_usage(df_new)

#         # predict user segment
#         df_user_segment = predict_user_segment(df_incr_usage)

#         #predict avg data after upgrade
#         df_avg_data_after = predict_avg_data_after(df_user_segment)

#         #predict recharge growth
#         df_recharge_growth = predict_recharge_growth(df_avg_data_after)

#         return df_recharge_growth
    
#     except Exception as e:  
#         raise RuntimeError(f'Error in handle_multi_user: {e}')

# def handle_direct_input(df_input): 
#     """
#     We're expecting a dataframe wich contains calculated informations of 1 or multiple users.
#     then we just predict the remaining features
#     """
#     print('Handling direct input')

#     try:
#         features = {
#             'data_flag',
#             'avg_data_before_upgrade',
#             'std_flag',
#             'std_before',
#             'total_recharge_before',
#             'recharge_flag'
#         }

#         missing = set(features) - set(df_input.columns)
#         if missing:
#             raise ValueError(
#                 f"handle_direct_input: missing columns {missing}. "
#                 f"Expected: {features}, got: {set(df_input.columns)}"
#             )
        
#         # Ensure correct column order and types
#         X = df_input[list(features)].copy()
#         X = X.apply(pd.to_numeric, errors='coerce')

#         # predict increased usage
#         df_incr_usage = predict_increased_usage(X)

#         # predict user segment
#         df_user_segment = predict_user_segment(df_incr_usage)
        
#         # predict avg data after upgrade
#         df_avg_data_after = predict_avg_data_after(df_user_segment)
        
#         # predict recharge growth  
#         df_recharge_growth = predict_recharge_growth(df_avg_data_after)

#         return df_recharge_growth
    
#     except Exception as e:
#         raise RuntimeError(f'Error in handle_direct_input: {e}')

# def predict_user_segment(df: pd.DataFrame) -> pd.DataFrame:

#     features = [
#         'avg_data_before_upgrade', 
#         'std_before', 
#         'total_recharge_before', 
#         'increased_usage',
#         'data_flag',
#         'std_flag',
#         'recharge_flag',
#         'upgrade_prob_confidence'
#     ]

#     missing = set(features) - set(df.columns)
#     if missing:
#         raise ValueError(
#             f"predict_user_segment: missing columns {missing}. "
#             f"Expected: {features}, got: {set(df.columns)}"
#         )
#     # Ensure correct column order and types
#     X = df[list(features)].copy()
#     X = X.apply(pd.to_numeric, errors='coerce')
    
#     y_pred = user_seg_model.predict(X)
#     proba = user_seg_model.predict_proba(X)


#     X['user_segment'] = y_pred
    
#     X['user_segment_prob_confidence'] = proba[np.arange(len(y_pred)), y_pred]

#     # Add user column back if present in input
#     if 'user' in df.columns:
#         X['user'] = df['user'].values

#     return X

# def predict_recharge_growth(df: pd.DataFrame) -> pd.DataFrame:
#     try:
#         features = [
#             'avg_data_after_upgrade',
#             'avg_data_before_upgrade',
#             'increased_usage',
#             'std_before',
#             'total_recharge_before',
#             'user_segment'
#         ]

#         missing = set(features) - set(df.columns)
#         if missing:
#             raise ValueError(
#                 f"predict_recharge_growth: missing columns {missing}. "
#                 f"Expected: {features}, got: {set(df.columns)}"
#             )
        
#         # Ensure correct column order and types
#         X = df[list(features)].copy()
#         X = X.apply(pd.to_numeric, errors='coerce')

#         y_pred = rech_grow_model.predict(X)
#         X['recharge_growth'] = y_pred

#         # Add user column back if present in input
#         if 'user' in df.columns:
#             X['user'] = df['user'].values

#         return X
    
#     except Exception as e:
#         raise RuntimeError(f'Error in predict_recharge_growth: {e}')

# def predict_avg_data_after(df: pd.DataFrame) -> pd.DataFrame:
#     try:

#         features = [
#             'avg_data_before_upgrade',
#             'std_before',
#             'total_recharge_before',
#             'increased_usage',
#             'data_flag',
#             'std_flag',
#             'recharge_flag',
#             'upgrade_prob_confidence',
#             'user_segment',
#             'user_segment_prob_confidence'
#         ]

#         missing = set(features) - set(df.columns)
#         if missing:
#             raise ValueError(
#                 f"predict_avg_data_after: missing columns {missing}. "
#                 f"Expected: {features}, got: {set(df.columns)}"
#             )
        
#         # Ensure correct column order and types
#         X = df[list(features)].copy()
#         # X = X.apply(pd.to_numeric, errors='coerce')

#         X['user_segment'] = X['user_segment'].astype('category')

#         y_pred = avg_model.predict(X)
        
#         X['avg_data_after_upgrade'] = y_pred

#         # Add user column back if present in input
#         if 'user' in df.columns:
#             X['user'] = df['user'].values

#         return X
    
#     except Exception as e:
#         raise RuntimeError(f'Error in predict_avg_data_after: {e}')

# def predict_increased_usage(input_df: pd.DataFrame) -> pd.DataFrame:
#     try:

#         features = [
#             'avg_data_before_upgrade',
#             'std_before',
#             'total_recharge_before',
#             'data_flag',
#             'std_flag',
#             'recharge_flag'
#         ]

#         missing = set(features) - set(input_df.columns)
#         if missing:
#             raise ValueError(
#                 f"predict_increased_usage: missing columns {missing}. "
#                 f"Expected: {features}, got: {set(input_df.columns)}"
#             )

#         # Ensure correct column order and types
#         X = input_df[list(features)].copy()
#         X = X.apply(pd.to_numeric, errors='coerce')
        
#         y_pred = incr_usage_model.predict(X)
#         y_pred_prob = incr_usage_model.predict_proba(X)[:, 1]

#         X['increased_usage'] = y_pred 
#         X['upgrade_prob_confidence'] = y_pred_prob

#         # Add user column back if present in input
#         if 'user' in input_df.columns:
#             X['user'] = input_df['user'].values
#         return X
#     except Exception as e:
#         raise RuntimeError(f'Error in predict_increased_usage: {e}')

# def clean_and_process(df: pd.DataFrame) -> pd.DataFrame:
#     try:
#         features_expected = {
#             'dates',
#             'Phone Number',
#             'total_reloads',
#             'total_reload_amount',
#             'device_category',
#             'data_kb'
#         }

#         missing = features_expected - set(df.columns)
#         if missing:
#             raise ValueError(
#                 f"clean_and_process: missing columns {missing}. "
#                 f"Expected: {features_expected}, got: {set(df.columns)}"
#             )
        
#         # Impute the total_reload with the mean
#         if int(df['total_reload_amount'].isnull().sum()) > 0:
#             m = df[(df['total_reloads'] == 1) & (df['total_reload_amount'].notna())]['total_reload_amount'].mean()
#             df = df.copy()
#             df['total_reload_amount'] = df['total_reload_amount'].fillna(m)
#             print(f"Imputed missing values in total_reload_amount using mean: {m:.2f}")

#         # Check required device categories
#         df = df[df['device_category'] != '-'].copy()
#         df['device_category'] = df['device_category'].astype(int)

#         dev_expected = {5, 7, 4}
#         present_cats = set(df['device_category'].unique())
#         if not present_cats & dev_expected:
#             raise ValueError(f"No expected device categories found in device_category column. Found: {present_cats}")

#         # If 5 (smartphone) is present, remove all rows where device_category == 5
#         if 5 in present_cats:
#             df = df[df['device_category'] != 5]
#             print("Removed all rows where device_category == 5 (smartphone).")

#         # Ensure at least one row with device_category in {7, 4} (feature phones)
#         if not df['device_category'].isin([7, 4]).any():
#             raise ValueError("No rows with device_category 7 or 4 (feature phones) found after filtering.")
        
#         # convert dates columns to datetime
#         df['dates'] = pd.to_datetime(df['dates'], format='%Y%m%d')

#         # Sort the DataFrame by 'Phone Number' and 'dates'
#         df = df.sort_values(by=['Phone Number', 'dates'])

#         #we need to rename the 'Phone Number' column to 'user' in the original DataFrame
#         df = df.rename(columns={'Phone Number': 'user'})

#         #Let's get users who used feature/basic phone before
#         df = df[df['device_category'].isin([4,7])]

#         return df

#     except Exception as e:
#         raise RuntimeError(f'Error in clean_and_process: {e}')  

# def generate_features(df: pd.DataFrame ) -> pd.DataFrame:
#     try: 
#         features_expected = {
#             'dates',
#             'user',
#             'total_reloads',
#             'total_reload_amount',
#             'device_category',
#             'data_kb'
#         }

#         missing = features_expected - set(df.columns)
#         if missing:
#             raise ValueError(
#                 f"generate_features: missing columns {missing}. "
#                 f"Expected: {features_expected}, got: {set(df.columns)}"
#             )
        
#         # Group and aggregate
#         if 'user' not in df.columns:
#             raise ValueError("generate_features: 'user' column missing after preprocessing.")
        
#         # Check for empty DataFrame
#         if df.empty:
#             raise ValueError("generate_features: Input DataFrame is empty.")

#         # Ensure correct types for calculations
#         for col in ['total_reloads', 'total_reload_amount', 'device_category', 'data_kb']:
#             if col in df.columns:
#                 if col == 'device_category':
#                     df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
#                 else:
#                     df[col] = pd.to_numeric(df[col], errors='coerce')

#         # Aggregate usage stats per user for both periods
#         df = df.groupby('user').agg(
#             avg_data_before_upgrade=('data_kb', 'mean'),
#             std_before=('data_kb', 'std'),
#             total_recharge_before=('total_reload_amount', 'sum')
#         ).reset_index()

#         # Handle Missing values
#         df['data_flag'] = df['avg_data_before_upgrade'].isna().astype(int)
#         df['std_flag'] = df['std_before'].isna().astype(int)
#         df['recharge_flag'] = df['total_recharge_before'].isna().astype(int)

#         # Fill missing values with 0
#         df['avg_data_before_upgrade'] = df['avg_data_before_upgrade'].fillna(0)
#         df['std_before'] = df['std_before'].fillna(0)
#         df['total_recharge_before'] = df['total_recharge_before'].fillna(0)

#         return df
#     except Exception as e:
#         raise RuntimeError(f'Error in generate_features: {e}')  
