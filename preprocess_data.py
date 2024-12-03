from clearml import Dataset, Task, TaskTypes
import os
from pathlib import Path
import pandas as pd
import global_config
from sklearn.model_selection import train_test_split


task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='Pipeline step 2 preprocess dataset',
    task_type=TaskTypes.data_processing,
    reuse_last_task_id=False,
    tags=["Kaggle"]
)

# program arguments
# Use either dataset_task_id to point to a tasks artifact or
# use a direct url with dataset_url
args = {
    'dataset_id': '2e6550820ebe414cb42e983413830e25',
    'random_state': 777,
    'test_size': 0.2,
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
    print(f"Using dataset ID: {dataset.id}")
    df = pd.read_csv((Path(local_folder) / 'titanic_dataset.csv'))
else:
    raise ValueError("Missing dataset id")

# Clean up df a little bit
train, test = train_test_split(df, random_state=args["random_state"])
train, val = train_test_split(train, random_state=args["random_state"])

print("Train shape is", train.shape)
print("Validation shape is", val.shape)
print("Test shape is", test.shape)

train_nan_mask = train["Age"].isna()
train.loc[train_nan_mask, "Age"] = train["Age"].mean()

val_nan_mask = val["Age"].isna()
val.loc[val_nan_mask, "Age"] = train["Age"].mean()

test_nan_mask = test["Age"].isna()
test.loc[test_nan_mask, "Age"] = train["Age"].mean()

train.drop(["PassengerId", "Name", "SibSp", "Ticket", "Cabin", "Embarked", "Sex"], inplace=True, axis=1)
val.drop(["PassengerId", "Name", "SibSp", "Ticket", "Cabin", "Embarked", "Sex"], inplace=True, axis=1)
test.drop(["PassengerId", "Name", "SibSp", "Ticket", "Cabin", "Embarked", "Sex"], inplace=True, axis=1)

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
    dataset_name='Titanic Kaggle Preprocessed Dataset',
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
