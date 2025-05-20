import random

def recommend_mtn_offer(row):
    """
    Rule-based MTN offer recommendation with human-like, personalized reasons.
    Expects columns: user, user_segment, avg_data_before_upgrade, increased_usage, recharge_growth, etc.
    """
    user = row.get('user', 'this user')
    segment = str(row.get('user_segment', 'non')).lower()
    avg_data = row.get('avg_data_before_upgrade', 0)
    increased_usage = row.get('increased_usage', 0)
    recharge_growth = row.get('recharge_growth', 0)

    # Default values
    offer = "MTN Learn"
    reason = f"{user} is recommended to learn more about MTN data offers."

    # --- Rule-based logic ---
    if segment == 'high':
        offer = "MTN Irekure 30GB at 10k"
        reason = (
            f"User {user} is a heavy data user with strong activity on a feature phone. "
            "When upgraded to a smartphone, they are expected to increase their usage even more. "
            "We recommend the MTN Irekure 30GB bundle at 10k RWF for a full month to match their high demand. "
            "Dial *345*1*3# to buy."
        )
    elif segment == 'medium':
        offer = random.choice(["MTN Yolo 2GBs daily at 5k", "Gwamon 7GBs + 30SMS at 1k"])
        if offer.startswith("MTN Yolo"):
            reason = (
                f"User {user} shows consistent data usage and is likely to benefit from affordable daily bundles. "
                "The MTN Yolo 2GB daily at 5k RWF is a great fit for their growing needs."
            )
        else:
            reason = (
                f"User {user} is steadily increasing their data usage. "
                "A weekly bundle like Gwamon 7GBs + 30SMS at 1k RWF will support their transition to higher usage."
            )
    elif segment == 'low':
        offer = random.choice([
            "MTN Yolo Squad 1.5GB at 500",
            "MoMo Buy Danta bundles",
            "MTN MoMo app buy bundles"
        ])
        if offer == "MTN Yolo Squad 1.5GB at 500":
            reason = (
                f"User {user} is starting to use more data."
                "A low-cost daily bundle like Yolo Squad 1.5GB at 500 RWF will encourage more frequent usage."
            )
        elif offer == "MoMo Buy Danta bundles":
            reason = (
                f"User {user} is showing interest in data but hasn't fully adopted regular usage. "
                "Suggesting MoMo Buy Danta bundles campaign can help them discover affordable options."
            )
        else:
            reason = (
                f"User {user} may prefer convenience. "
                "Encourage them to use the MTN MoMo app to buy bundles easily without dialing USSD codes."
            )
    elif segment == 'non':
        offer = random.choice([
            "FREE 100 MBs",
            "MTN Learn (How to use data)",
            "MTN Ihereze (recharging loans)"
        ])
        if offer == "FREE 100 MBs":
            reason = (
                f"User {user} is not currently using data. "
                "Offering FREE 100 MBs can motivate them to try internet services on their device."
            )
        elif offer == "MTN Learn (How to use data)":
            reason = (
                f"User {user} has little or no data activity. "
                "Send them a guide on how to use data and set up their internet to help them get started."
            )
        else:
            reason = (
                f"User {user} may need a little push to start using data. "
                "Offering a recharge loan (MTN Ihereze) can help them get connected even if they have no balance."
            )
    else:
        offer = "MTN Learn"
        reason = (
            f"User {user}'s usage pattern is unclear. "
            "Recommend learning about MTN's data offers and how to get started."
        )

    # Add extra info for expected growth if available
    if increased_usage and increased_usage > 0.7:
        reason += " They are expected to significantly increase their data usage after upgrading."
    elif increased_usage == 0:
        reason += " They are unlikely to use much data unless incentivized."

    # Add recharge growth context if available
    if recharge_growth and recharge_growth > 0.5:
        reason += " Recharge activity also suggests readiness for bigger bundles."

    return {
        "offer": offer,
        "reason": reason
    }