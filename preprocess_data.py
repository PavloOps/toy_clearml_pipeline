import os
from pathlib import Path
import pandas as pd
from clearml import Dataset, Task
import global_config
from sklearn.model_selection import train_test_split


task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='preprocess data',
    task_type='data_processing',
    reuse_last_task_id=False
)

# Create the folder we'll output the preprocessed data into
preprocessed_data_folder = Path('/tmp')
if not os.path.exists(preprocessed_data_folder):
    os.makedirs(preprocessed_data_folder)

# Get the dataset
dataset = Dataset.get(
    dataset_project=global_config.PROJECT_NAME,
    dataset_name='titanic_dataset',
)
local_folder = dataset.get_local_copy()
print(f"Using dataset ID: {dataset.id}")

# Clean up the data a little bit
df = pd.read_csv((Path(local_folder) / 'titanic_dataset.csv'))


train, test = train_test_split(df, random_state=42)
train, val = train_test_split(df, random_state=42)

print("Train shape is", train.shape)
print("Validation shape is", val.shape)
print("Test shape is", test.shape)


train_nan_mask = train["Age"].isna()
train.loc[train_nan_mask, "Age"] = train["Age"].mean()

val_nan_mask = val["Age"].isna()
val.loc[val_nan_mask, "Age"] = train["Age"].mean()

test_nan_mask = test["Age"].isna()
test.loc[test_nan_mask, "Age"] = train["Age"].mean()

train.drop(["PassengerId", "Name", "SibSp", "Ticket", "Cabin", "Embarked"], inplace=True, axis=1)
val.drop(["PassengerId", "Name", "SibSp", "Ticket", "Cabin", "Embarked"], inplace=True, axis=1)
test.drop(["PassengerId", "Name", "SibSp", "Ticket", "Cabin", "Embarked"], inplace=True, axis=1)

train.to_csv(path_or_buf=preprocessed_data_folder / 'train.csv')
print(f"Preprocessed train")
print(train.head())

val.to_csv(path_or_buf=preprocessed_data_folder / 'val.csv')
print(f"Preprocessed val")
print(val.head())

test.to_csv(path_or_buf=preprocessed_data_folder / 'test.csv')
print(f"Preprocessed test")
print(test.head())


# Create a new version of the dataset, which is cleaned up
new_dataset = Dataset.create(
    dataset_project=dataset.project,
    dataset_name='preprocessed_titanic_dataset',
    parent_datasets=[dataset]
)
new_dataset.add_files(preprocessed_data_folder / 'train.csv')
new_dataset.add_files(preprocessed_data_folder / 'val.csv')
new_dataset.add_files(preprocessed_data_folder / 'test.csv')

new_dataset.get_logger().report_table(title='Train data', series='head', table_plot=train.head())
new_dataset.get_logger().report_table(title='Validation data', series='head', table_plot=val.head())
new_dataset.get_logger().report_table(title='Test data', series='head', table_plot=test.head())

new_dataset.finalize(auto_upload=True)

# Log to console which dataset ID was created
print(f"Created preprocessed dataset with ID: {new_dataset.id}")
