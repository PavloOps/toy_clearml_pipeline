from clearml.automation import TaskScheduler


scheduler = TaskScheduler()

scheduler.add_task(
    schedule_task_id='d2b5f4e09ea442d7aa3853c3cf821c41',
    queue='default',
    minute = 1
)

scheduler.start_remotely(queue='services')
