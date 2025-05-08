import numpy as np
import pandas as pd

def inject_non_adopters(df, user_col='Phone Number', date_col='dates', usage_col='data_kb', count=1000):
    """
    Inject fake non-adopters into the dataset by setting their data_kb to 0 after upgrade.

    Parameters:
    - df: Raw synthetic DataFrame
    - user_col: Column representing user IDs (default: 'Phone Number')
    - date_col: Column representing dates (format: YYYYMMDD)
    - usage_col: Column representing data usage (default: 'data_kb')
    - count: Number of users to modify

    Returns:
    - Modified DataFrame with ~count non-adopters
    """
    df = df.copy()
    
    # Ensure date column is integer and sorted
    df[date_col] = df[date_col].astype(int)
    df = df.sort_values([user_col, date_col])

    # Step 1: Get users with usage before upgrade
    users_with_usage = df[df[usage_col] > 1000][user_col].unique()

    # Step 2: Pick random subset to convert into non-adopters
    if len(users_with_usage) < count:
        raise ValueError(f"Only {len(users_with_usage)} users found with usage. Can't select {count}.")
    selected_users = np.random.choice(users_with_usage, size=count, replace=False)

    # Step 3: Get each selected user's upgrade date
    upgrade_dates = df.groupby(user_col)[date_col].min().to_dict()

    # Step 4: Set data_kb to 0 after upgrade
    for user in selected_users:
        upgrade_date = upgrade_dates[user]
        mask = (df[user_col] == user) & (df[date_col] > upgrade_date)
        df.loc[mask, usage_col] = 0.0

    return df
