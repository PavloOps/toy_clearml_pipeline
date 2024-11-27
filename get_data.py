import global_config
from clearml import Task, Dataset, StorageManager
import pandas as pd

task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='get data',
    task_type='data_processing',
    reuse_last_task_id=False
)

task.connect(global_config)

# Get the data and a path to the file
manager = StorageManager()
dataset_path = f"./{global_config.FILE_NAME}"
df = pd.read_csv(dataset_path)
print(f"Dataset path: {dataset_path}")
print(df.head())

# Create a ClearML dataset
dataset = Dataset.create(
    dataset_name='titanic_dataset',
    dataset_project=global_config.PROJECT_NAME
)
# Add the local files we downloaded earlier
dataset.add_files(dataset_path)

# Let's add graphs as statistics in the plots section!
dataset.get_logger().report_table(title='Titanic Data', series='head', table_plot=df.head())

# Finalize and upload the data and labels of the dataset
dataset.finalize(auto_upload=True)

print(f"Created dataset with ID: {dataset.id}")
print(f"Data size: {len(df)}")
