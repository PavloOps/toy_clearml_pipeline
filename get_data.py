from clearml import Task, Dataset, StorageManager, TaskTypes
import global_config
import pandas as pd

task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='Pipeline step 1 get csv dataset',
    task_type=TaskTypes.data_processing,
    reuse_last_task_id=False,
    tags=["Kaggle"]
)

# only create the task, we will actually execute it later; makes DRAFT mode
task.execute_remotely()

task.connect(global_config)

# Get the data and a path to the file
manager = StorageManager()
dataset_path = f"./{global_config.FILE_NAME}"
df = pd.read_csv(dataset_path)
print(f"Dataset path: {dataset_path}")
print(df.head())

# Create a ClearML dataset
dataset = Dataset.create(
    dataset_name='Titanic Kaggle Dataset',
    dataset_project=global_config.PROJECT_NAME,
)
# Add the local files we downloaded earlier
dataset.add_files(dataset_path)

# Let's add graphs as statistics in the plots section!
dataset.get_logger().report_table(
    title='Titanic Data',
    series='head',
    table_plot=df.head()
)

# Finalize and upload the data and labels of the dataset
dataset.finalize(auto_upload=True)

print(f"Created dataset with ID: {dataset.id}")
print(f"Data size: {len(df)}")

# we are done
print('Done')
