from clearml import Task, Dataset, TaskTypes
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import global_config
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name="Pipeline step 3 train model",
    tags=["Kaggle", "Titanic", "Holdout", "Catboost"],
    task_type=TaskTypes.training
)

# Arguments
args = {
    'dataset_id': '72cee4435af3475e864e33f1a075ac58'
}
task.connect(args)

# only create the task, we will actually execute it later; makes DRAFT mode
# task.execute_remotely()

print('Retrieving Titanic Dataset')

if args['dataset_id']:
    dataset = Dataset.get(dataset_id=args['dataset_id'])
    local_folder = dataset.get_local_copy()
    print(f"Using dataset ID: {dataset.id}")

    train = pd.read_csv((Path(local_folder) / 'train.csv'))
    X_train = train.drop("Survived", axis=1)
    y_train = train["Survived"]

    val = pd.read_csv((Path(local_folder) / 'val.csv'))
    X_val = val.drop("Survived", axis=1)
    y_val = val["Survived"]
    print('Dataset is loaded')
else:
    raise ValueError("Missing dataset id")

model = CatBoostClassifier(allow_writing_files=False, random_seed=777, verbose=True)
model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=50)
print("Best validation score:", model.best_score_)
print("Iterations' amount:", model.tree_count_)

model.save_model("titainc_model.cbm")
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

print('Done')
task.close()
