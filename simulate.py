import pandas as pd
import numpy as np


def simulate_surgeries(n):

    df = pd.DataFrame()

    # age - normal dist
    age = np.floor(np.random.normal(50, 15, n))
    age = np.clip(age, 10, 100)
    df['age'] = age
    df['age'] = df['age'].astype(int)

    # gender - uniform
    df['gender'] = np.random.choice(['m', 'f'], n)

    # race - from SG's demography
    p_c = 0.743
    p_m = 0.135
    p_i = 0.09
    p_o = 1 - (p_c + p_m + p_i)
    df['race'] = np.random.choice(['c', 'm', 'i', 'o'], n, p=[p_c, p_m, p_i, p_o])

    # marital status - from SG's demography (https://www.singstat.gov.sg/find-data/search-by-theme/population/marital-status-marriages-and-divorces/latest-data)
    p = [0.297, 0.612, 0.047, 0.043]
    p = np.array(p)/np.sum(p)
    df['marital_status'] = np.random.choice(['single', 'married', 'widowed', 'separated/divorced'], n, p=p)

    # surgery complexity - exponential dist
    complexities = np.arange(10) + 1
    l = 0.25
    p = np.exp(-l*complexities)
    p = p/p.sum()
    sc = np.random.choice(complexities, n, p=list(p))
    df['surgery_complexity'] = sc

    # specialty - from reference above (DOI:10.1503/cjs.008119)
    specialties = ['cardiac', 'dental', 'general', 'gynecology', 'neurosurgery', 'orthopedic', 'otolaryngology', 'plastic', 'thoracic', 'urology', 'vascular']
    p = [9.6, 0.7, 14, 17.2, 6.1, 22.9, 2, 4, 2.7, 14.5, 6.3]
    p = np.array(p)/np.sum(p)
    df['specialties'] = np.random.choice(specialties, n, p=list(p))

    # scheduled date - for simplicity, hospital's operating time is 24 h, and schedule collision is allowed
    def random_dates(start, end, n=10):
        start_u = start.value//10**9
        end_u = end.value//10**9
        return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

    start = pd.to_datetime('2021-01-01')
    end = pd.to_datetime('2023-12-31')
    df['scheduled_date'] = random_dates(start, end, n)

    # Other categorical features
    n_cat_features = 12
    n_cats = 5
    for i in range(1, n_cat_features+1):
        cats = [chr(x) for x in range(65, 65 + n_cats)]
        df[f"categorical_{i}"] = np.random.choice(cats, n)


    # CANCELLATION PROBABILITY

    # formulating probability of cancellation - individual contribution is between 0-1
    df_prob = pd.DataFrame()

    # contribution from age - assumption: cancellation rate increases linearly with age
    df_prob['c_age'] = df['age'] / df['age'].max()

    # contribution from gender - male more likely to cancel (https://doi.org/10.1016/j.colegn.2023.03.013)
    df_prob['c_gender'] = df['gender'].apply(lambda x: 1 if x=='m' else 0)

    # other contribution
    # TODO

    # label generation
    global_surgery_cancellation_rate = 0.15
    c_total = df_prob.sum(axis=1)
    p_unscaled = c_total/c_total.max()
    scaling_factor = global_surgery_cancellation_rate / p_unscaled.mean()
    p_scaled = p_unscaled * scaling_factor
    y = p_scaled.apply(lambda x: 1 if np.random.rand()<x else 0)

    df['is_cancelled'] = y

    return df
