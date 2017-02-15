"""
Main init for the Package.
In this module all the modules will be exposed and initialized.abs
"""
# pylint: disable=invalid-name

# Variable to define what model to import.
# This could be useful to import additional modules depending on requirements
# or OS specifications.

mode = 1

if mode == 1:
    print("Mode 1 succesfully imported")
    from .mode1 import *
else:
    print("Mode 2 succesfully imported")
    from .mode2 import *


