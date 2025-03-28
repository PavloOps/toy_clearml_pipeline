from clearml import Task, TaskTypes
import psycopg2

connection_params = {
    "host": "TO BE OVERRIDE",
    "user": "TO BE OVERRIDE",
    "password": "TO BE OVERRIDE",
    "database": "TO BE OVERRIDE"
}

task = Task.init(
    project_name="TEST_PIPE_FROM_TASKS",
    task_name="Step 2",
    task_type=TaskTypes.data_processing
)

task.connect(name='Database', mutable=connection_params)
print('Database Arguments: {}'.format(connection_params))

task.execute_remotely()

print("Hello from task 2")
