from clearml import PipelineController, TaskTypes


# We will use the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
def step_one(query, connection_params, filename):
    print("Start function step_one")
    # make sure we have scikit-learn for this step, we need it to use to unpickle the object
    from clearml import Dataset
    import pandas as pd
    import psycopg2
    from json import loads

    connection_params = loads(connection_params)
    print(type(connection_params))

    connection = psycopg2.connect(**connection_params)
    with connection.cursor() as cursor:
        cursor.execute(query)
        df = pd.DataFrame(
            cursor.fetchall(),
            columns=[desc[0] for desc in cursor.description],
        )
        print(df.head())

    # Create a ClearML dataset
    dataset = Dataset.create(
        dataset_name='Titanic Raw Dataset from Pipeline (SQL)',
        dataset_project='Titanic Pet Project',
    )
    df.to_csv(filename)
    dataset.add_files(filename)

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

    return dataset_id, connection_params


# We will use the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
def step_two(dataset_id, filename, connection_params, test_size=0.2, random_state=42):
    from clearml import Dataset
    import os
    from pathlib import Path
    import pandas as pd
    from sklearn.model_selection import train_test_split

    dataset = Dataset.get(dataset_id=dataset_id)
    local_folder = dataset.get_local_copy()
    print(f"Using dataset ID: {dataset.id}")
    df = pd.read_csv((Path(local_folder) / filename), index_col=0)

    train, test = train_test_split(df, random_state=random_state, test_size=test_size)
    train, val = train_test_split(train, random_state=random_state, test_size=test_size)

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

    return processed_dataset_id, connection_params


# We will use the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
def step_three(processed_dataset_id, connection_params):
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

    train = pd.read_csv((Path(local_folder) / 'train.csv'), index_col=0)
    X_train = train.drop(["survived", "passengerid"], axis=1)
    y_train = train["survived"]

    val = pd.read_csv((Path(local_folder) / 'val.csv'), index_col=0)
    X_val = val.drop(["survived", "passengerid"], axis=1)
    y_val = val["survived"]
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

    return model, processed_dataset_id, connection_params


def step_four(model, processed_dataset_id, connection_params):
    import psycopg2
    from clearml import Task, TaskTypes, Dataset
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from catboost import CatBoostClassifier
    from tqdm import tqdm

    dataset = Dataset.get(dataset_id=processed_dataset_id)
    local_folder = dataset.get_local_copy()
    print(f"Using dataset ID: {dataset.id}")

    test = pd.read_csv((Path(local_folder) / 'test.csv'), index_col=0)
    X_test = test.drop(["survived", "passengerid"], axis=1)
    X_test.dropna(inplace=True)
    y_true = test["survived"]

    y_pred = model.predict_proba(X_test)
    probas_for_survived = y_pred[:, 1]
    np.savez("probabilities.npz", probabilities=probas_for_survived)
    test['probas'] = probas_for_survived

    print("We are here: ", type(connection_params))
    print(connection_params)

    connection = psycopg2.connect(**connection_params)
    with connection.cursor() as cursor:
        for _, line in tqdm(test[['passengerid', 'probas']].iterrows(), desc="Titanic Probabilities"):
            passenger_id, probability = line

            cursor.execute(
                """
                    INSERT INTO toy_example.survive(
                    passengerid,
                    probability)
                    VALUES(%s, %s);
                """,
                (passenger_id, probability),
            )

    connection.commit()
    return probas_for_survived


if __name__ == '__main__':
    # create the pipeline controller
    pipe = PipelineController(
        project='Titanic Pet Project',
        name='Titanic Pipeline Demonstration',
        version='1.0',
        output_uri="http://192.168.1.3:8081"
    )

    # set the default execution queue to be used (per step we can override the execution)
    pipe.set_default_execution_queue('default')

    # add pipeline components
    pipe.add_parameter(
        'filename',
        r"C:\Users\analytic_ko\PycharmProjects\pythonProject\titanic_dataset.csv"
    )

    pipe.add_parameter(
        'connection_params',
        {"user": "postgres", "database": "postgres"}
    )

    pipe.add_parameter(
        'query',
        'SELECT * FROM toy_example.titanic_raw_data'
    )

    print(pipe.get_parameters())
    print("step1 added")

    pipe.add_function_step(
        name='step_one',
        function=step_one,
        function_kwargs=dict(
            filename='${pipeline.filename}',
            query='${pipeline.query}',
            connection_params='${pipeline.connection_params}'
        ),
        function_return=['dataset_id', 'connection_params'],
        cache_executed_step=True,
        execution_queue='default',
        task_type=TaskTypes.data_processing
    )
    pipe.add_function_step(
        name='step_two',
        # parents=['step_one'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=step_two,
        function_kwargs=dict(
            dataset_id='${step_one.dataset_id}',
            filename='${pipeline.filename}',
            connection_params='${step_one.connection_params}'
        ),
        function_return=['processed_dataset_id', 'connection_params'],
        cache_executed_step=True,
        execution_queue='default',
        task_type=TaskTypes.data_processing
    )
    print("step2 added")
    pipe.add_function_step(
        name='step_three',
        # parents=['step_two'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=step_three,
        function_kwargs=dict(
            processed_dataset_id='${step_two.processed_dataset_id}',
            connection_params='${step_two.connection_params}'
        ),
        function_return=['model', 'processed_dataset_id', 'connection_params'],
        # cache_executed_step=True,
        execution_queue='default',
        task_type=TaskTypes.training
    )
    print("step3 added")

    pipe.add_function_step(
        name='step_four',
        function=step_four,
        function_kwargs=dict(
            model='${step_three.model}',
            processed_dataset_id='${step_three.processed_dataset_id}',
            connection_params='${step_three.connection_params}',
        ),
        function_return=['probas_for_survived'],
        execution_queue='default',
        task_type=TaskTypes.inference
    )

    # For debugging purposes run on the pipeline on current machine
    # Use run_pipeline_steps_locally=True to further execute the pipeline component Tasks as subprocesses.
    # pipe.start_locally(run_pipeline_steps_locally=True)

    # Start the pipeline on the services queue (remote machine, default on the clearml-server)
    pipe.start(queue='default')

    print('Pipeline is completed!')
