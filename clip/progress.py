""" A customized progress bar. """
import math

import progressbar

def custom_progressbar(task, steps):
    """A context manager that provides a progress bar.

    :param task: A short string identifying the process whose progress is being
            shown.
    :param steps: An integer or float, indicating the number of steps in the
            process.

    When progress occurs, pass the current step number to the `update()` method
    of the returned object.
    """
    digits = int(math.log10(steps))+1
    widgets = [
        '|',
        f'{task:^25s}',
        ' ',
        progressbar.Bar(),
        progressbar.Percentage(),
        '| ',
        progressbar.SimpleProgress(format=f'%(value_s){digits}s/%(max_value_s){digits}s'),
        ' |',
        progressbar.ETA(
            format_not_started='',
            format_finished='%(elapsed)8s',
            format='%(eta)8s',
            format_zero='',
            format_NA=''
        ),
        '|'
    ]
    return progressbar.ProgressBar(max_value=steps, widgets=widgets)
