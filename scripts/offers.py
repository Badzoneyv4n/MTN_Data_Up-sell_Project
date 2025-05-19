def recommend_mtn_offer(row):
    if row['user_segment'] == 'non':
        if row['recharge_growth'] < 2000:
            return 'MTN Learn'
        else:
            return 'FREE 100MB'

    elif row['user_segment'] == 'low':
        if row['recharge_growth'] < 5000:
            return 'Yolo Squad 1.5GB'
        else:
            return 'Gwamon 7GB + 30SMS'

    elif row['user_segment'] == 'medium':
        if row['increased_usage'] == 1:
            return 'Yolo 2GB Daily at 5k'
        else:
            return 'MoMo Buy Danta Bundles'

    elif row['user_segment'] == 'high':
        if row['recharge_growth'] > 15000:
            return 'MTN Irekure 30GB at 10k'
        else:
            return 'MTN MIFI'

    return 'MTN Ihereze'  # default if nothing matches
