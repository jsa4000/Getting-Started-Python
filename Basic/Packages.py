"""
This is a Package sample
"""
# pylint: disable=invalid-name
import Package.SubPackage2 as pkg2
import Package.SubPackage1 as pkg1

# When a Package is created it's usesful to have a setup.py that will contains
# all the dependences needed. This will be the installator for the package.
# You can see more in detail the content of this file in the following link:
#   http://www.diveintopython3.net/packaging.html

# The package must have the following strucuture
#   someFolder
#   |-- string_func
#   |   |-- __init__.py
#   |   |-- stringToUpper.py
#   |   |-- stringToLower.py
#   |   `-- strengthLength.py
#   `-- example1.py

# NOTE: The process to create packages in Python 2 and Python 3 is different.
# Something to need be aware it's using relative and global imports.
#  - Relative Import (dot define the module to import). (This means the current folder or PATH.
#       from .Car import Car
#  - Global Import
#       from Car import Car


# Let's create an object from a custom Package created.
car = pkg1.Car("MiCoche")

print(car._wheels)
print(car.wheels)
print(car.name)

# Cannot set attribute since there is no setter created for the Property
#car.name = "Property being Modified"

# Following two ways to modify variables.abs
# THe fist one it's allowed but it'snot recommended
car.__name = "Private Memeber being Modified"
car.wheels = 16

print(car._wheels)
print(car.wheels)
print(car.name)


# Call directly to the imported functions in SubPackage2
# In this case we have only the sum function
print(pkg2.sum(2,3))



