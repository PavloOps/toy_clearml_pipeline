from clearml.automation import PipelineController


def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    print("Cloning Task id={} with parameters: {}".format(a_node.base_task_id, current_param_override))
    return True


def post_execute_callback_example(a_pipeline, a_node):
    print("Completed Task id={}".format(a_node.executed))


if __name__ == '__main__':
    pipe = PipelineController(
        name="Test Pipeline",
        project="TEST_PIPE_FROM_TASKS",
        version="0.0.1",
        add_pipeline_tags=False
    )

    pipe_params = {
        "user": "postgres",
        "password": "123",
        "database": "zdorovie_three_years",
        "host": "localhost"
    }   # добавила, чтобы посмотреть, как работает оверрайд, в коде просто принты

    for k, v in pipe_params.items():
        pipe.add_parameter(name=k, default=v)

    # pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="step1",
        base_task_project="TEST_PIPE_FROM_TASKS",
        base_task_name="Step 1",
        parameter_override={
            "Database/host": "${pipeline.host}",
            "Database/port": "${pipeline.port}",
            "Database/user": "${pipeline.user}",
            "Database/password": "${pipeline.password}",
            "Database/database": "${pipeline.database}",
        },
        execution_queue="default"
        # pre_execute_callback=pre_execute_callback_example,
        # post_execute_callback=post_execute_callback_example
    )

    pipe.add_step(
        name="step2",
        base_task_project="TEST_PIPE_FROM_TASKS",
        base_task_name="Step 2",
        parents=['step1'],
        parameter_override={
            "Database/host": "${pipeline.host}",
            "Database/port": "${pipeline.port}",
            "Database/user": "${pipeline.user}",
            "Database/password": "${pipeline.password}",
            "Database/database": "${pipeline.database}",
        },
        execution_queue="default"
        # pre_execute_callback=pre_execute_callback_example,
        # post_execute_callback=post_execute_callback_example
    )

    # pipe.start_locally()   # этот работает хорошо
    pipe.start()   # этот не работает =(
