
"""
It's needed to put some documentation using pyLint.
You can disable by using some directives as you can see some lines bellow-
"""
import time
# The Python programing language
# pylint: disable=invalid-name

#To run the code in Visual Studio Code Just press Ctrl+Shift+B

x = 1
y = 5

def my_sum(parm_x, param_y):
    """
    This is as Sum and a small doc string to show how to create a function in ython
    """
    return parm_x + param_y

#Following call will print the sume beween x and y
print(x+y)
#Following call will print the sume beween x and y by calling the function created
print("With functions", my_sum(x, y))


def my_sum_extended(param_x, param_y, param_z=None):
    """
    Another function that takes three parameters. One of them optional
    """
    if param_z is None:
        return param_x+param_y
    else:
        return param_x+param_y+param_z

#Following call will print the sume beween x and y by calling the function created
print("With functions", my_sum_extended(x, y))
#Create new variable
z = 4
#Following call will print the sume beween x and y by calling the function created
print("With functions", my_sum_extended(x, y, z))

#Also any function can be assigned to a variable
#Since the functions in Python are also considered objects
a = my_sum_extended
type(a)
# Now let's call tp the Function by using the variable that we have assigend to the Function
print("This is a call from a variable", a(x, y, 4))

# Lambda expresions
print("# LAMBDA (or Anonymous) expresions")

# This lambda expresion is used to simply the function.
def power(number, exp=2):
    """
    This return the power of the number by the given exponent
    """
    return number**2
power_lambda = lambda number, exp=2: number**exp
# Now see the differences betwen the two elements
print(power(3))
print(power_lambda(3))

# GENERATORS
print("# Generators")
# This is an advanced feature in Python
# This is a better way to return values from a big source
# This is used in case you have to read a file or a database so you don't want
# to store all the data in memory. This will allow the possibility to access
# to the data on-request instead getting all the values.
# An example could be data form database. (Remember DataSet or DataResult)
my_database = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#Not lets create a function that will iterate over those elements.
def get_item(database):
    """
        This will be a yield generator that loop over a list withot
        consuming memory or store inte the system.
    """
    for dbitem in database:
        yield dbitem

# After I have created the generator, now I can call the function to
# start getting the values.
for item in get_item(my_database):
    print(item)

# DECORATORS
print("# Decorators")

# Decorators are used to add functionality extra to the functions.
# In some cases it's necessary to add previous checkings to the data or
# some logging .

# A decoratos it's a functiona that it's called prior to another so:


# Following are the idea behidn decoratos
def my_function():
    """
    This is the main function that will be called
    """
    print("Process")
    return True

def my_decorator(func):
    """
        This is a decorator
    """
    print("Do something")
    return func()

#Let's call to the the function
my_decorator(my_function)


# So let's imagine we want to create a function that print the time that takes
# the execution of any function

def timeit(method):
    """
    This is time function.
    Inside there is another function that will be called to return the value of the inside function
    """
    def timed(*args, **kw):
        """
        This function is create internally.
        This can be sued to store local information of the main function.
        """
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print("{0} ({1}, {2}) {3} sec".format(method.__name__, args, kw, te-ts))
        return result
    return timed

# This is the wa the decorators are declared
@timeit
def some_func(some_a, some_b=3):
    """
    This is the main function
    """
    time.sleep(0.2)
    return some_a+some_b

#This is to get values from the function
my_value = some_func(4, 5)
print(my_value)

# Why using  a function inside a functions (Inner function):
# Here is the link https://realpython.com/blog/python/inner-functions-what-are-they-good-for/

#   - Encapsulation. Inner function are hidden from the global scope. This mean you cannot call
#     directly the function unless yuo first call the top function first.
#   - Multiple Inputs. A function tha receive any input whatever the type and are capable of
#     return the same value. An example of this could be a function that receieve wheter
#     a filename or a handle to read all the lines. The function automatically detect if it's
#     the name of a file, so open it and treat them as the same case.
#   - Closure: this is used to remember the state inside a function. This could be similar to the
#     way JavaScript manage the clases because a function at the end define initial
#     values befire it's called.

#An example of closure function
def generate_power(number):
    """
    This function will generate the power for the given number
    """
    def nth_power(power):
        """
        This is the enclusure function.
        This will return a number that is store at creation time. like a initialization
        """
        return number**power
    return nth_power

# Initialize the value with 3
mypowr_func = generate_power(3)
# now every time I call to the function the number that will be used will be 3
print(mypowr_func(2))
print(mypowr_func(2))
print(mypowr_func(2))

# Now lets see some helpuf options
# Map, reduce and Filter






