
import pandas as pd
import numpy as np

# Mapping of employment length to numeric values
emp_length_mapping = {
    '10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6,
    '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1,
    '< 1 year': 0.5, 'n/a': 0
}

# Mapping of U.S. states to broader regions
state_to_region = {
    'CA': 'West', 'OR': 'West', 'UT': 'West', 'WA': 'West', 'CO': 'West',
    'NV': 'West', 'AK': 'West', 'MT': 'West', 'HI': 'West', 'WY': 'West', 'ID': 'West',
    'AZ': 'SouthWest', 'TX': 'SouthWest', 'NM': 'SouthWest', 'OK': 'SouthWest',
    'GA': 'SouthEast', 'NC': 'SouthEast', 'VA': 'SouthEast', 'FL': 'SouthEast',
    'KY': 'SouthEast', 'SC': 'SouthEast', 'LA': 'SouthEast', 'AL': 'SouthEast',
    'WV': 'SouthEast', 'DC': 'SouthEast', 'AR': 'SouthEast', 'DE': 'SouthEast',
    'MS': 'SouthEast', 'TN': 'SouthEast',
    'IL': 'MidWest', 'MO': 'MidWest', 'MN': 'MidWest', 'OH': 'MidWest',
    'WI': 'MidWest', 'KS': 'MidWest', 'MI': 'MidWest', 'SD': 'MidWest',
    'IA': 'MidWest', 'NE': 'MidWest', 'IN': 'MidWest', 'ND': 'MidWest',
    'CT': 'NorthEast', 'NY': 'NorthEast', 'PA': 'NorthEast', 'NJ': 'NorthEast',
    'RI': 'NorthEast', 'MA': 'NorthEast', 'MD': 'NorthEast', 'VT': 'NorthEast',
    'NH': 'NorthEast', 'ME': 'NorthEast'
}

def clean_loan_data(df):
    df = df.copy()

    # Filter loan statuses
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    bad_statuses = [
        "Charged Off", "Default",
        "Does not meet the credit policy. Status:Charged Off",
        "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"
    ]
    df['target'] = df['loan_status'].apply(lambda x: 1 if x in bad_statuses else 0).astype(int)

    # Map employment length and region
    df['emp_length_int'] = df['emp_length'].map(emp_length_mapping)
    df['region'] = df['addr_state'].map(state_to_region)

    # Drop useless columns
    drop_cols = ['id', 'member_id', 'desc', 'url', 'title', 'zip_code', 'policy_code']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    return df
