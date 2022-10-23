# Taken from the mlsuite library

import sys


class TeeFile:
    """Writes to multiple files at once"""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)

    def flush(self):
        for file in self.files:
            file.flush()


def is_interactive_shell() -> bool:
    return sys.__stdin__.isatty()

# from contextlib import redirect_stdout, redirect_stderr
# with open(arguments.options.output_file, 'a') as out, open(arguments.options.error_file, 'a') as err:
#     if arguments.options.verbose and is_interactive_shell():
#         out_file = TeeFile(out, sys.stdout)
#         err_file = TeeFile(err, sys.stderr)
#     else:
#         out_file, err_file = out, err
#
#     with redirect_stdout(out_file), redirect_stderr(err_file):
#         func(arguments)
