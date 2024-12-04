from clearml import Task, TaskTypes, Dataset
from pathlib import Path
import global_config
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(
    project_name=global_config.PROJECT_NAME,
    tags=["Kaggle"],
    task_type=TaskTypes.inference,
    task_name="Pipeline step 4 get probabilities on new data"
)

args = {
    'training_task_id': '3e2f363bf5044f13a5c1febfb1afc992',
    'dataset_id': '72cee4435af3475e864e33f1a075ac58'
}

# store arguments, later we will be able to change them from outside the code; makes DRAFT mode
task.connect(args)
print('Arguments: {}'.format(args))

# only create the task, we will actually execute it later
# task.execute_remotely()

# get dataset from task's artifact
if args['training_task_id']:
    training_upload_task = Task.get_task(task_id=args['training_task_id'])
    print('Input task id={} artifacts {}'.format(args['training_task_id'], list(training_upload_task.artifacts.keys())))
    output_model = training_upload_task.models['output']
    model_path = output_model[0].get_local_copy()
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
else:
    raise ValueError("Missing training task id")

if args['dataset_id']:
    dataset = Dataset.get(dataset_id=args['dataset_id'])
    local_folder = dataset.get_local_copy()
    print(f"Using dataset ID: {dataset.id}")

    test = pd.read_csv((Path(local_folder) / 'test.csv'))
    X_test = test.drop("Survived", axis=1)
    X_test.dropna(inplace=True)
    y_true = test["Survived"]
else:
    raise ValueError("Missing dataset id")

# get probabilities
y_pred = loaded_model.predict_proba(X_test)
probas_for_survived = y_pred[:, 1]
np.savez("probabilities.npz", probabilities=probas_for_survived)

task.upload_artifact('probas_for_survived', artifact_object=probas_for_survived)

print('Uploading artifacts in the background')

# we are done
print('Done')
task.close()
