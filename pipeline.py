from clearml import PipelineController


# We will use the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
def step_one(file_name):
    # make sure we have scikit-learn for this step, we need it to use to unpickle the object
    from clearml import Dataset
    import pandas as pd

    # Get the data and a path to the file
    df = pd.read_csv(file_name)
    print(f"Dataset file: {file_name}")
    print(df.head())

    # Create a ClearML dataset
    dataset = Dataset.create(
        dataset_name='Titanic Raw Dataset from Pipeline',
        dataset_project='Titanic Pet Project',
    )
    # Add the local files we downloaded earlier
    dataset.add_files(file_name)

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
    dataset_id = dataset.id

    return dataset_id


# We will use the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
def step_two(dataset_id, test_size=0.2, random_state=42):
    from clearml import Dataset
    import os
    from pathlib import Path
    import pandas as pd
    from sklearn.model_selection import train_test_split


    dataset = Dataset.get(dataset_id=dataset_id)
    local_folder = dataset.get_local_copy()
    print(f"Using dataset ID: {dataset.id}")
    df = pd.read_csv((Path(local_folder) / 'titanic_dataset.csv'))

    train, test = train_test_split(df, random_state=random_state, test_size=test_size)
    train, val = train_test_split(train, random_state=random_state, test_size=test_size)

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
        dataset_project='Titanic Pet Project',
        dataset_name='Titanic Preprocessed Dataset From Pipeline',
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
    processed_dataset_id = new_dataset.id

    return processed_dataset_id


# We will use the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
def step_three(processed_dataset_id):
    from clearml import Dataset
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    from catboost import CatBoostClassifier
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    dataset = Dataset.get(dataset_id=processed_dataset_id)
    local_folder = dataset.get_local_copy()
    print(f"Using dataset ID: {dataset.id}")

    train = pd.read_csv((Path(local_folder) / 'train.csv'))
    X_train = train.drop("Survived", axis=1)
    y_train = train["Survived"]

    val = pd.read_csv((Path(local_folder) / 'val.csv'))
    X_val = val.drop("Survived", axis=1)
    y_val = val["Survived"]
    print('Dataset is loaded')

    model = CatBoostClassifier(allow_writing_files=False, random_seed=777, verbose=True)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=50)
    print("Best validation score:", model.best_score_)
    print("Iterations' amount:", model.tree_count_)

    model.save_model("titainc_model_from_pipeline.cbm")
    print('Model is trained & stored')

    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    return model


if __name__ == '__main__':

    # create the pipeline controller
    pipe = PipelineController(
        project='Titanic Pet Project',
        name='Pipeline demo',
        version='1.0'
    )

    # set the default execution queue to be used (per step we can override the execution)
    pipe.set_default_execution_queue('default')

    # add pipeline components
    pipe.add_parameter(
        name='filename',
        description='file name to load',
        default='/home/pavloops/PycharmProjects/example/titanic_dataset.csv'
    )
    print("step1 added")
    pipe.add_function_step(
        name='step_one',
        function=step_one,
        function_kwargs=dict(file_name='${pipeline.filename}'),
        function_return=['dataset_id'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='step_two',
        # parents=['step_one'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=step_two,
        function_kwargs=dict(dataset_id='${step_one.dataset_id}'),
        function_return=['processed_dataset_id'],
        cache_executed_step=True,
    )
    print("step2 added")
    pipe.add_function_step(
        name='step_three',
        # parents=['step_two'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=step_three,
        function_kwargs=dict(processed_dataset_id='${step_two.processed_dataset_id}'),
        function_return=['model'],
        cache_executed_step=True,
    )
    print("step3 added")

    # For debugging purposes run on the pipeline on current machine
    # Use run_pipeline_steps_locally=True to further execute the pipeline component Tasks as subprocesses.
    # pipe.start_locally(run_pipeline_steps_locally=True)

    # Start the pipeline on the services queue (remote machine, default on the clearml-server)
    pipe.start(queue='default')

    print('pipeline completed')