"""
This allows to the Package only to expose the modules that are needed

There are several ways to import a Package. By using relative and global imports.

- Relative Import (dot define the module to import). This means the current folder or PATH.
 
from .Car import Car

- Global Import

 from Car import Car

"""
from .Car import Car


