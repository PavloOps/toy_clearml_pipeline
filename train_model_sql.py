from clearml import Task, Dataset, TaskTypes, OutputModel
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

global_config = {
    'PROJECT_NAME': 'Titanic Toy Example (SQL)',
    'FILENAME': 'titanic_dataset_from_query.csv'
}

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(
    project_name=global_config["PROJECT_NAME"],
    task_name="Pipeline step 3 train model (SQL)",
    tags=["SQL", "Kaggle", "Titanic", "Holdout", "Catboost"],
    task_type=TaskTypes.training,
    output_uri="http://0.0.0.0:8081"
)

# Arguments
args = {
    'dataset_id': 'c38d57c32e2b469a8fb8e2db2801622b'
}
task.connect(args)

# only create the task, we will actually execute it later; makes DRAFT mode
task.execute_remotely()

print('Retrieving Titanic Dataset')

if args['dataset_id']:
    dataset = Dataset.get(dataset_id=args['dataset_id'])
    local_folder = dataset.get_local_copy()
    print(f"Using dataset ID: {dataset.id}")

    train = pd.read_csv((Path(local_folder) / 'train.csv'), index_col=0)
    X_train = train.drop(["survived", "passengerid"], axis=1)
    y_train = train["survived"]

    val = pd.read_csv((Path(local_folder) / 'val.csv'), index_col=0)
    X_val = val.drop(["survived", "passengerid"], axis=1)
    y_val = val["survived"]
    print('Dataset is loaded')
else:
    raise ValueError("Missing dataset id")

model = CatBoostClassifier(allow_writing_files=False, random_seed=742, verbose=True)
model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, early_stopping_rounds=50)
print("Best validation score:", model.best_score_)
print("Iterations' amount:", model.tree_count_)

# Сначала сохраняем модель локально
local_model_path = "titainc_model.cbm"
model.save_model(local_model_path)

print("Current working directory:", os.getcwd())
print("Model saved at:", local_model_path)

# # Связываем сохранённую модель с ClearML и загружаем её
# output_model = OutputModel(task=task)
# output_model.update_weights(
#     weights_filename=local_model_path,  # Локальный файл
#     target_filename="titanic_model.cbm"  # Имя файла на сервере
# )
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
