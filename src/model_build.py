import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

def setup(name: str = 'app_logger', log_file: str = 'app.log') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file)
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

jeo_cols = ['age', 'fnlwgt', 'hours.per.week', 'capital_loss']

category_cols = [
    'workclass', 'marital_status', 'occupation',
    'relationship', 'sex', 'native_country'
]
numerical_cols = [
    'age', 'fnlwgt', 'education.num', 'capital_loss',
    'hours.per.week', 'education_encoded',
    'capital-gain-flag', 'capital-gain-log'
]

def data_load(url):
    logger.info("Data Load Started")
    try:
        df = pd.read_csv(url)
        logger.info(f"Data Load Successfully | shape={df.shape}")
        return df
    except Exception as e:
        logger.error(f"Data Load Failed: {e}")
        raise

def data_split(df):
    logger.info("Data Splitting Start")
    X = df.drop(columns=['income'])
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    y_train = y_train.astype(int)
    y_test  = y_test.astype(int)
    logger.info(f"Data Splitting Successfully | train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test

def procesor(category_cols, numerical_cols, jeo_cols):
    logger.info("Making a Processor Started")
    preprocessor = ColumnTransformer(
        [
            ('Scaler', StandardScaler(), numerical_cols),
            ('jeo', PowerTransformer(method='yeo-johnson'), jeo_cols),
            ('ohc', OneHotEncoder(handle_unknown='ignore', sparse_output=False), category_cols)
        ]
    )
    logger.info("Making a Processor Successfully")
    return preprocessor

def model_build_pipe(preprocessor, X_train, X_test, y_train, y_test):
    logger.info("Model Pipeline Build Started")
    best_pipe = ImbPipeline(
        [
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('model', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                max_depth=5, subsample=0.8, random_state=42
            ))
        ]
    )
    logger.info("Model Fitting Started")
    best_pipe.fit(X_train, y_train)
    logger.info("Model Fitting Successfully")

    logger.info("Model Prediction Started")
    y_pred = best_pipe.predict(X_test)
    logger.info("Model Prediction Successfully")

    report = classification_report(y_test, y_pred)
    print(report)
    logger.debug(f"Classification Report:\n{report}")

    return best_pipe

def save_data(data_path, pipe):
    logger.info("Model Save Start")
    save_path = os.path.join(data_path, "model_evalution")
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "best_pipe.pkl")
    joblib.dump(pipe, model_path)
    logger.info(f"Model Saved Successfully | path={model_path}")

def main():
    logger.info("Main Pipeline Execution Started")
    df = data_load(r'data\cleaning\df_clean.csv')
    X_train, X_test, y_train, y_test = data_split(df)
    preprocessor = procesor(category_cols, numerical_cols, jeo_cols)
    final_pipe = model_build_pipe(preprocessor, X_train, X_test, y_train, y_test)
    save_data("data", final_pipe)
    logger.info(" Main Pipeline Execution Completed")

if __name__ == "__main__":
    main()

