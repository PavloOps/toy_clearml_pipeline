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
    task_name="Step 1",
    task_type=TaskTypes.data_processing
)

task.connect(name='Database', mutable=connection_params)
print('Database Arguments: {}'.format(connection_params))

task.execute_remotely()

print("Hello from task 1")
with psycopg2.connect(**connection_params) as connection:
    with connection.cursor() as cursor:
        cursor.execute('''
        SELECT COUNT(*)
        FROM reference_data.product p''')
        result = cursor.fetchone()
print(result)
print("Done!")