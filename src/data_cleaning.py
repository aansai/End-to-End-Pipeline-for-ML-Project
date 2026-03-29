import numpy as np
import pandas as pd
import os
import logging

def setup(name: str = 'app_logger',log_file: str = 'app.log') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    file_handler  = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup()
logger.debug("Logger initialized and configuration loaded.")
logger.info("Starting the Data Preprocessing Pipeline...")
logger.warning("External data source detected: checking file integrity.")
logger.error("Data validation failed: missing columns in input CSV.")
logger.critical("Memory limit reached: process terminated.")

def data_load(url):
    logger.info("Data Load Started") 
    df = pd.read_csv(url)
    logger.info("Data Load Successfully")
    return df

def Workclass(df):
    logger.info('Cleaning of Workclass Start')
    df['workclass'] = df['workclass'].replace('?', 'Unknown')
    df['workclass'] = df['workclass'].replace({
        'State-gov': 'Government',
        'Federal-gov': 'Government',
        'Local-gov': 'Government',
        'Self-emp-not-inc': 'Self-Employed',
        'Self-emp-inc': 'Self-Employed',
        'Without-pay': 'Unemployed',
        'Never-worked': 'Unemployed',
        '?': 'Unknown'
    })
    logger.info("Workclass Columns Cleaning Completed")
    return df

def Education(df):
    logger.info('Cleaning of education column Start')
    df['education'] = df['education'].replace({
        'Preschool': 'Below-HS',
        '1st-4th': 'Below-HS',
        '5th-6th': 'Below-HS',
        '7th-8th': 'Below-HS',
        '9th': 'Some-HS',
        '10th': 'Some-HS',
        '11th': 'Some-HS',
        '12th': 'Some-HS',
        'HS-grad': 'HS-grad',
        'Some-college': 'Some-college',
        'Assoc-acdm': 'Associates',
        'Assoc-voc': 'Associates',
        'Bachelors': 'Bachelors',
        'Masters': 'Masters',
        'Prof-school': 'Prof-school',
        'Doctorate': 'Doctorate',
    })
    logger.info("Education column Cleaning Successfully")
    return df

education_order = {
    'Below-HS': 0,
    'Some-HS': 1,
    'HS-grad': 2,
    'Some-college': 3,
    'Associates': 4,
    'Bachelors': 5,
    'Masters': 6,
    'Prof-school': 7,
    'Doctorate': 8
}

def education_or(education_order, df):
    logger.info('Education Order Column Created')
    df['education_encoded'] = df['education'].map(education_order)
    logger.info('Education Column Drop')
    df.drop(columns=['education'], inplace=True)
    return df

def Marital(df):
    logger.info('marital_status column Cleaning Start')
    df['marital.status'] = df['marital.status'].replace({
        'Married-civ-spouse': 'Married',
        'Married-spouse-absent': 'Married',
        'Married-AF-spouse': 'Married',
        'Never-married': 'Never-married',
        'Divorced': 'Divorced',
        'Separated': 'Separated',
        'Widowed': 'Widowed',
    })
    df = df.rename(columns={'marital.status': 'marital_status'})
    logger.info('marital_status cleaning Successfully')
    return df

def Occupation(df):
    logger.info("Occupation Columns Cleaning Started")
    df['occupation'] = df['occupation'].replace('?', 'Unknown')
    df['occupation'] = df['occupation'].replace({
        'Exec-managerial': 'White-Collar',
        'Prof-specialty': 'White-Collar',
        'Adm-clerical': 'White-Collar',
        'Tech-support': 'White-Collar',
        'Sales': 'White-Collar',
        'Craft-repair': 'Blue-Collar',
        'Machine-op-inspct': 'Blue-Collar',
        'Transport-moving': 'Blue-Collar',
        'Handlers-cleaners': 'Blue-Collar',
        'Farming-fishing': 'Blue-Collar',
        'Other-service': 'Service',
        'Priv-house-serv': 'Service',
        'Protective-serv': 'Service',
        'Armed-Forces': 'Special',
        'Unknown': 'Unknown',
    })
    logger.info("Occupation Column Cleaning Successfully")
    return df

def Relationship(df):
    logger.info("Relationship column Cleaning Started")
    df['relationship'] = df['relationship'].replace({
        'Husband': 'Married',
        'Wife': 'Married',
        'Not-in-family': 'Alone',
        'Unmarried': 'Alone',
        'Own-child': 'Dependent',
        'Other-relative': 'Dependent',
    })
    logger.info('Relationship columns Cleaning Successfully')
    return df

def drop_race(df):
    df.drop(columns=['race'], inplace=True)
    logger.info('Race Column Drop Successfully')
    return df

def Native(df):
    logger.info('native_country cleaning Started')
    df['native.country'] = df['native.country'].replace({
        'United-States': 'North-America',
        'Canada': 'North-America',
        'Mexico': 'North-America',
        'Puerto-Rico': 'North-America',
        'Outlying-US(Guam-USVI-etc)': 'North-America',
        'Honduras': 'Central-America',
        'Nicaragua': 'Central-America',
        'El-Salvador': 'Central-America',
        'Guatemala': 'Central-America',
        'Dominican-Republic': 'Central-America',
        'Haiti': 'Central-America',
        'Cuba': 'Central-America',
        'Jamaica': 'Central-America',
        'Trinadad&Tobago': 'Central-America',
        'Columbia': 'South-America',
        'Ecuador': 'South-America',
        'Peru': 'South-America',
        'England': 'Europe',
        'Germany': 'Europe',
        'Greece': 'Europe',
        'Italy': 'Europe',
        'Poland': 'Europe',
        'Portugal': 'Europe',
        'Ireland': 'Europe',
        'France': 'Europe',
        'Hungary': 'Europe',
        'Scotland': 'Europe',
        'Yugoslavia': 'Europe',
        'Holand-Netherlands': 'Europe',
        'India': 'Asia',
        'China': 'Asia',
        'Japan': 'Asia',
        'Vietnam': 'Asia',
        'Taiwan': 'Asia',
        'Philippines': 'Asia',
        'Hong': 'Asia',
        'Cambodia': 'Asia',
        'Laos': 'Asia',
        'Thailand': 'Asia',
        'Iran': 'Asia',
        'South': 'Other',
    })
    df = df.rename(columns={'native.country': 'native_country'})
    logger.info('Native_country Cleaning Successfully')
    return df

def Income(df):
    df['income'] = df['income'].replace('<=50K', '0').replace('>50K', '1')
    logger.info('Change the Income in 0&1')
    return df

def capital(df):
    df = df.rename(columns={'capital.loss': 'capital_loss'})
    logger.info('Rename the capital.loss column Successfully')
    return df

def Gain(df):
    logger.info("Making a capital-gain-flag or capital-gain-log col by capital-gain ")
    df['capital-gain-flag'] = (df['capital.gain'] > 0).astype(int)
    df['capital-gain-log'] = np.log1p(df['capital.gain'])
    df.drop(columns=['capital.gain'], inplace=True)
    logger.info('making a columns successfully')
    return df

def save_data(data_path, df):
    logger.info("Data Save Start")
    save_path = os.path.join(data_path, "cleaning")
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, "df_clean.csv"), index=False)
    logger.info("Data Saved Successfully")

def main():
    df = data_load(r'data\raw\df_gath.csv') 
    df = Workclass(df)
    df = Education(df)
    df = education_or(education_order, df)
    df = Marital(df)
    df = Occupation(df)
    df = Relationship(df)
    df = drop_race(df)
    df = Native(df)
    df = Income(df)
    df = capital(df)
    df = Gain(df)
    save_data("data", df)

if __name__ == "__main__":
    main()