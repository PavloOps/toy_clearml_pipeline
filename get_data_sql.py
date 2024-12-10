import psycopg2
from clearml import Task, Dataset, StorageManager, TaskTypes
import pandas as pd

CONNECTION_PARAMS = {
    "user": "postgres",
    "database": "postgres"
}

global_config = {
    'PROJECT_NAME': 'Titanic Toy Example (SQL)',
    'QUERY': 'SELECT * FROM toy_example.titanic_raw_data',
    'FILENAME': '"titanic_dataset_from_query.csv"'
}

task = Task.init(
    project_name=global_config["PROJECT_NAME"],
    task_name='Pipeline step 1 get dataset by SQL-query',
    task_type=TaskTypes.data_processing,
    reuse_last_task_id=False,
    tags=["SQL"]
)

# only create the task, we will actually execute it later; makes DRAFT mode
task.execute_remotely()

task.connect(global_config)
task.connect(CONNECTION_PARAMS)

# Get the data and a path to the file
with psycopg2.connect(**CONNECTION_PARAMS) as conn:
    with conn.cursor() as cursor:
        cursor.execute(global_config['QUERY'])
        df = pd.DataFrame(
            cursor.fetchall(),
            columns=[desc[0] for desc in cursor.description],
        )
        print(df.head())


df.to_csv(global_config['FILENAME'])

# Create a ClearML dataset
dataset = Dataset.create(
    dataset_name='Titanic Kaggle Dataset (SQL)',
    dataset_project=global_config["PROJECT_NAME"],
)
# Add the local files we downloaded earlier
dataset.add_files(global_config['FILENAME'])

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
