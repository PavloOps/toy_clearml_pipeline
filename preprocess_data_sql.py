from clearml import Dataset, Task, TaskTypes
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


global_config = {
    'PROJECT_NAME': 'Titanic Toy Example (SQL)',
    'FILENAME': 'titanic_dataset_from_query.csv'
}


task = Task.init(
    project_name=global_config["PROJECT_NAME"],
    task_name='Pipeline step 2 preprocess dataset (SQL)',
    task_type=TaskTypes.data_processing,
    reuse_last_task_id=False,
    tags=["SQL"]
)

# program arguments
# Use either dataset_task_id to point to a tasks artifact or
# use a direct url with dataset_url
args = {
    'dataset_id': 'db389d8afc9b4f929b71ad5aff23de2d',
    'random_state': 742,
    'test_size': 0.1,
}

# store arguments, later we will be able to change them from outside the code
task.connect(args)
print('Arguments: {}'.format(args))

# only create the task, we will actually execute it later; makes DRAFT mode
task.execute_remotely()

# get dataset from dataset id
if args['dataset_id']:
    dataset = Dataset.get(dataset_id=args['dataset_id'])
    local_folder = dataset.get_local_copy()
    df = pd.read_csv((Path(local_folder) / global_config['FILENAME']))
else:
    raise ValueError("Missing dataset id")

# Clean up df a little bit
train, test = train_test_split(df, random_state=args["random_state"])
train, val = train_test_split(train, random_state=args["random_state"])

print("Train shape is", train.shape)
print("Validation shape is", val.shape)
print("Test shape is", test.shape)

train_nan_mask = train["age"].isna()
train.loc[train_nan_mask, "age"] = train["age"].mean()

val_nan_mask = val["age"].isna()
val.loc[val_nan_mask, "age"] = train["age"].mean()

test_nan_mask = test["age"].isna()
test.loc[test_nan_mask, "age"] = train["age"].mean()

train.drop(["name", "sibsp", "ticket", "cabin", "embarked", "sex"], inplace=True, axis=1)
val.drop(["name", "sibsp", "ticket", "cabin", "embarked", "sex"], inplace=True, axis=1)
test.drop(["name", "sibsp", "ticket", "cabin", "embarked", "sex"], inplace=True, axis=1)

# Create the folder we'll output the preprocessed data into
preprocessed_data_folder = Path('/tmp')
if not os.path.exists(preprocessed_data_folder):
    os.makedirs(preprocessed_data_folder)

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
    dataset_name='Titanic Kaggle Preprocessed Dataset (SQL)',
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
print('Done')
