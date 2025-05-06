import random
import pandas as pd
import numpy as np

def generate_unique_phone_numbers(count=20000, prefix='2257987'):
    """
    Generate a list of unique phone numbers with a fixed prefix.

    Parameters:
    - count: Total numbers to generate
    - prefix: String prefix for each phone number (e.g., '25078898')

    Returns:
    - List of unique phone numbers as strings
    """
    suffix_length = 12 - len(prefix)
    max_suffix = 10**suffix_length
    if count > max_suffix:
        raise ValueError(f"Cannot generate {count} unique numbers with prefix '{prefix}'")

    suffixes = random.sample(range(max_suffix), count)
    phone_numbers = [prefix + str(suffix).zfill(suffix_length) for suffix in suffixes]
    return phone_numbers

def generate_user_dates(phone_numbers, min_days=50 , max_days=250 , start_date='2024-09-01', end_date='2025-04-21'):
    """
    Generate a list of rows with Phone Number and random dates per user.

    Parameters:
    - phone_numbers: list of phone numbers
    - min_days: minimum number of records per user
    - max_days: maximum number of records per user
    - start_date, end_date: range of dates to sample from

    Returns:
    - A DataFrame with 'Phone Number' and 'date' columns
    """
    records=[]
    date_range = pd.date_range(start=start_date, end=end_date).tolist()

    for number in phone_numbers:
        num_records = random.randint(min_days, max_days)
        dates = random.choices(date_range, k=num_records)

        for date in dates:
            records.append({'Phone Number': number, 'dates': date.strftime('%Y%m%d')})

    return pd.DataFrame(records)

def generate_total_reloads(n_rows):
    """
    Generate a skewed distribution of total_reloads values.

    Parameters:
    - n_rows: Number of rows to generate

    Returns:
    - A list of integers representing total daily reloads
    """
    values = [1]*60 + [2]*25 + [3]*8 + [4, 5, 6]*2 + list(range(7, 13))
    
    return random.choices(values, k=n_rows)


def generate_total_reload_amount(n_rows, nan_ratio=0.007):
    """
    Generate total reload amounts using a skewed distribution with NaNs.

    Parameters:
    - n_rows: Total rows to generate
    - nan_ratio: Ratio of NaN values (e.g. 0.007 = 0.7%)

    Returns:
    - A list of floats with NaNs injected
    """
    # Generate skewed reload amounts
    reloads = np.random.lognormal(mean=5, sigma=1.2, size=n_rows)  # lognormal is skewed
    reloads = np.clip(reloads, 2.0, 40000)  # clamp to min/max
    reloads = np.round(reloads, 2)  # round to 2 decimal places

    # Inject NaNs
    n_nans = int(n_rows * nan_ratio)
    nan_indices = random.sample(range(n_rows), n_nans)
    for idx in nan_indices:
        reloads[idx] = np.nan

    return reloads.tolist()

def generate_imei(n_rows):
    """
    Generate unique 10-digit IMEIs starting with 1 or 2.

    Parameters:
    - n_rows: Total IMEIs to generate

    Returns:
    - List of unique 10-digit IMEIs as strings
    """
    imeis = set()

    while len(imeis) < n_rows:
        prefix = str(random.choice([1, 2]))
        suffix = ''.join(random.choices('0123456789', k=9))
        imei = prefix + suffix
        imeis.add(imei)
    return list(imeis)


def generate_brand_and_model(n_rows):
    """
    Generate brand_name and model_name columns with realistic skew.

    Parameters:
    - n_rows: Total rows to generate

    Returns:
    - Two lists: brand_names, model_names
    """
    
    brands = [f"Brand_{i+1}" for i in range(49)]

    weights = [random.randint(1, 50) for _ in brands]

    total_weight = sum(weights)

    probs = [w / total_weight for w in weights]

    # Assign 3–10 models per brand
    brand_model_map = {
        brand: [f"{brand}_Model_{j+1}" for j in range(random.randint(3, 10))]
        for brand in brands
    }

    brand_list = random.choices(brands, weights=probs, k=n_rows)
    model_list = [random.choice(brand_model_map[brand]) for brand in brand_list]

    return brand_list, model_list

def generate_device_category(n_rows):
    """
    Generate device categories using realistic distribution from MTN data.

    Parameters:
    - n_rows: Number of device entries

    Returns:
    - List of device_category integers
    """
    categories = [5, 7, 4, 2, 3]  # Smartphone, Basic, FeaturePhone, Modem, M2M
    weights =   [423, 318, 239, 0.3, 0.1]  # relative weight, normalized internally
    total = sum(weights)
    probs = [w / total for w in weights]

    return random.choices(categories, weights=probs, k=n_rows)

def generate_data_kb(n_rows, zero_ratio=0.5):
    """
    Generate data_kb values with a heavy right-skew and many zeros.

    Parameters:
    - n_rows: Total number of entries
    - zero_ratio: Fraction of rows with 0 usage (e.g., 0.07 = 7%)

    Returns:
    - List of floats
    """
    n_zeros = int(n_rows * zero_ratio)
    n_nonzero = n_rows - n_zeros

    # Log-normal for non-zero values (skewed usage)
    data_vals = np.random.lognormal(mean=12.5, sigma=1.2, size=n_nonzero)
    data_vals = np.clip(data_vals, 0.5, 8_000_000)
    data_vals = np.round(data_vals, 2)

    # Add zeros
    full_data = list(data_vals) + [0.0] * n_zeros
    random.shuffle(full_data)

    return full_data

