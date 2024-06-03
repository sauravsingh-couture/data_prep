import sys
from ai.couture.obelisk.commons.utils.ArgumentsParser import ArgumentsParser

# from assortment.utils.ydata_profiler import YDataProfiler
from assortment.utils.test import Test



class MainClass(object):
    def __init__(self, sys_arg):
        self.sys_arg = sys_arg
        return

    def main(self):
        """instantiate object for global arguments parser"""
        args = ArgumentsParser()
        """parse task from args."""
        task = args.parse_task(self.sys_arg)

        demand_forecast_module = {
            # "ydata_profiler": YDataProfiler,
            "test": Test,
        }

        modules = {}
        modules.update(demand_forecast_module)

        tasks_list = list(modules.keys())
        if task in tasks_list:
            """ Instantiate object for a task and run it """
            task_obj = modules[task]()
            task_obj.execute(self.sys_arg)
        else:
            print("KeyError: \'{}\' No such task found. \n"
                  "List of tasks available : \n{}".format(task, tasks_list))
            sys.exit(1)


if __name__ == '__main__':
    """instantiate object for all use-cases and call the main function."""
    usecase_obj = MainClass(sys.argv[1])
    usecase_obj.main()
