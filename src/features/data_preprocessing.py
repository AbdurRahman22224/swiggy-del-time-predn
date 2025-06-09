import pandas as pd
import logging
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, 
    MinMaxScaler, 
    OrdinalEncoder)
import joblib
from sklearn import set_config

# set the transformer outputs to pandas
set_config(transform_output='pandas')

# columns to preprocess in data

num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]

nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

target_col = "time_taken"

# generate order for ordinal encoding

traffic_order = ["low","medium","high","jam"]

distance_type_order = ["short","medium","long","very_long"]

# create logger
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    
    except FileNotFoundError:
        logger.error("The file to load does not exist")
    
    return df


def drop_missing_values(data: pd.DataFrame) -> pd.DataFrame:

    logger.info(f"The original dataset with missing values has {data.shape[0]} rows and {data.shape[1]} columns")
    df_dropped = data.dropna()
    logger.info(f"The dataset with missing values dropped has {df_dropped.shape[0]} rows and {df_dropped.shape[1]} columns")
    missing_vals = df_dropped.isna().sum().sum()
    
    if missing_vals > 0:
        raise ValueError("The dataframe has missing values")
    return df_dropped


def save_transformer(transformer, save_dir: Path, transformer_name: str):
    # form the save location
    save_location = save_dir / transformer_name
    # save the transformer
    joblib.dump(value=transformer,filename=save_location)
    
def train_preprocessor(preprocessor, data: pd.DataFrame):
    # fit on the data
    preprocessor.fit(data)
    return preprocessor

def perform_transformations(preprocessor, data: pd.DataFrame):
    # transform the data
    transformed_data = preprocessor.transform(data)
    return transformed_data

def save_data(data: pd.DataFrame, save_path: Path) -> None:
    data.to_csv(save_path, index=False)
    

def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def join_X_and_y(X: pd.DataFrame, y: pd.Series):
    # join based on indexes
    joined_df = X.join(y,how='inner')
    return joined_df
    
    
if __name__ == "__main__":
    # paths
    # root path
    # root path
    root_path = Path(__file__).parent.parent.parent
    # data load path
    train_data_path = root_path / "data" / "interim" / "train.csv"
    test_data_path = root_path / "data" / "interim" / "test.csv"
    # save data directory
    save_data_dir = root_path / "data" / "processed"
    # make dir if not preseny
    save_data_dir.mkdir(exist_ok=True,parents=True)
    # train and test data save paths
    # filenames
    train_trans_filename = "train_trans.csv"
    test_trans_filename = "test_trans.csv"
    # save path for train and test
    save_train_trans_path = save_data_dir / train_trans_filename
    save_test_trans_path = save_data_dir / test_trans_filename
    
    # preprocessor
    preprocessor = ColumnTransformer(transformers=[
            ("scale", MinMaxScaler(), num_cols),
            ("nominal_encode", OneHotEncoder(drop="first",
                                            handle_unknown="ignore",
                                            sparse_output=False), nominal_cat_cols),
            ("ordinal_encode", OrdinalEncoder(categories=[traffic_order,
                                                          distance_type_order],
                                            encoded_missing_value=-999,
                                            handle_unknown="use_encoded_value",
                                            unknown_value=-1), ordinal_cat_cols)],
                                    remainder="passthrough",
                                    n_jobs=-1,
                                    force_int_remainder_cols=False,
                                    verbose_feature_names_out=False)
    
    
    # load the train and test data with missing values dropped
    train_df = drop_missing_values(load_data(data_path=train_data_path))
    logger.info("Train data loaded successfully")
    test_df = drop_missing_values(load_data(data_path=test_data_path))
    logger.info("Test data loaded successfully")
    
    # split the train and test data
    X_train, y_train = make_X_and_y(data=train_df,target_column=target_col)
    X_test, y_test = make_X_and_y(data=test_df, target_column=target_col)
    logger.info("Data splitting completed")
    
    # fit the preprocessor on X_train
    train_preprocessor(preprocessor=preprocessor, data=X_train)
    logger.info("Preprocessor is trained")
    
    # transform the data
    X_train_trans =  perform_transformations(preprocessor=preprocessor, data=X_train)
    logger.info("Train data is transformed")
    X_test_trans = perform_transformations(preprocessor=preprocessor, data=X_test)
    logger.info("Test data is transformed")
    
    # join back X and y
    train_trans_df = join_X_and_y(X_train_trans, y_train)
    test_trans_df = join_X_and_y(X_test_trans, y_test)
    logger.info("Datasets joined")
    
    # save the transformed data
    data_subsets = [train_trans_df, test_trans_df]
    data_paths = [save_train_trans_path,save_test_trans_path]
    filename_list = [train_trans_filename, test_trans_filename]
    for filename , path, data in zip(filename_list, data_paths, data_subsets):
        save_data(data=data, save_path=path)
        logger.info(f"{filename.replace(".csv","")} data saved to location")
        
    # save the preprocessor to location
    # transformer name
    transformer_filename = "preprocessor.joblib"
    # directory to save transformers
    transformer_save_dir = root_path / "models"
    transformer_save_dir.mkdir(exist_ok=True)
    # save the transformer
    save_transformer(transformer=preprocessor,
                     save_dir=transformer_save_dir,
                     transformer_name=transformer_filename)
    logger.info("Preprocessor saved to location")
    