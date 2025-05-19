def recommend(df_input, mode='auto'):
    """
    Main recommendation system entry point.

    Parameters:
    - df_input: pandas DataFrame of user data
    - mode: 'auto', 'single', 'multi', or 'direct'

    Returns:
    - DataFrame with prediction, segment, recommendation, and reason
    """
    # Automatically detect the input mode
    if mode == 'auto':
        mode = detect_input_mode(df_input)

    if mode == 'single':
        return handle_single_user(df_input)

    elif mode == 'multi':
        return handle_multi_user(df_input)

    elif mode == 'direct':
        return handle_direct_input(df_input)

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'single', 'multi', 'direct', or 'auto'.")

def detect_input_mode(df):
    """
    Determines the input type: direct features or raw usage logs (single or multi user).
    """
    if {'avg_data_before_upgrade', 'std_before', 'total_recharge_before'}.issubset(df.columns):
        return 'direct'
    
    if df['Phone Number'].nunique() == 1:
        return 'single'

    return 'multi'

def handle_single_user(df_input):
    pass

def handle_multi_user(df_input):
    pass

def handle_direct_input(df_input): 
    pass