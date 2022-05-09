def on_render():
    print('this runs before executing my_task')

def on_finish():
    print('this runs when my_task executes without errors!')

def on_failure():
    print('this runs when my_task raise an exception during execution!')