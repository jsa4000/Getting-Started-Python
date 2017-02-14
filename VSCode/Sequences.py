"""
This is a document to explain how the sequences works in python
"""
# pylint: disable=invalid-name

def my_sum(param_x, param_y):
    """
    This is a simple funciom
    """
    return param_x + param_y

print("# TYPES")

# Let's get started by seeing what it the type of a string
my_string = "This is a sequence"
print(type(my_string))

# What's the type of a None variable
print(type(None))
print(type(1))
print(type(my_sum))

# Now let's create SEQUENCES

print("# TUPLES")
# Tuples are inmutable,, This meanm the content cannot be changed.
# Tuples are created by using normal brackets  "(" and ")".
my_first_tuple = (1, 2, 3, 4, 200)
print(my_first_tuple)
print(type(my_first_tuple))

# You can mix different types within tuples
my_mixed_tuple = (1, 2, 3, 4, "Javier", None, {"Alonso":"Driver", "Javier":"Programmer"})
print(my_mixed_tuple)

# You can access to an index of the curent tuple by its index
print(my_mixed_tuple[3])
print(my_mixed_tuple[6]["Alonso"])

print("#Casting Tuple into list")
#Also you can cast the tuple to be a list, in order to be mutable.
#By casting the tuple into a list this will return a list
my_casting_tuple = list(my_first_tuple)
my_casting_tuple[2] = "Something New"
print(my_first_tuple)
print(my_casting_tuple)

print("# LIST")
# List are mutables
# List are created using Square-brackets "[" and "]"
my_first_list = [1, 2, 3, 4, 200]
print(my_first_list)
print(type(my_first_list))

# List can be modified in contrast to the tuples
my_first_list[2] = "Something New"
print(my_first_list)

# List can be access by using indexing
print(my_first_list[0])
print(my_first_list[2])

# Append values
my_first_list.append("Last Item")
print(my_first_list)

print("# DICTIONARY")
# Dictionary are mutables
# Dictionary are created using braces
# {"key1":"value1","key2":"value2",.. }
my_Dictionary = {"id_01":"Javier", "id_02":"Alvaro", "id_03":"Pedro"}
print(my_Dictionary)

# You can access to the items by using the key
print(my_Dictionary["id_01"])

# You can add a new item into the Dictionary
my_Dictionary["id_04"] = "Jacinto"
print(my_Dictionary)

# Dictionaty are not orderer as you can see when I have added the last item.

# Loop over items. This is by using the for sentence (for each)
print("# FOR LOOP OVER ITEMS")
for item in my_first_list:
    print(item)

# You can use an index to iterate over the items
print("# WHILE SENTENCE OVER ITEMS")
index = 0
while index < len(my_first_list):
    # print the current item b usign the index
    print("{0} : {1}".format(index, my_first_list[index]))
    # Move to the next item in the list
    index += 1

# Also, you can use enumerate function to append an index for each item like
print("# ENUMERATE OVER ITEMS")
for index, item in enumerate(my_first_list):
    print("{0} : {1}".format(index, item))

print("# Formatting literals")
# The format function is formatting pourposes.
# You can tweak some parameters and add format to the literals
name = "Javier"
surname = "Santos"
age = 22
height = 1.734334
my_string = "Hello, I'm {0} {1}. My height is {2} and I am {3:.2f} years old." \
    .format(name, surname, age, height)
# "\" is used in python to split lines too long using python.
print(my_string)


print("# Operations over lists")
my_list = [200, 2, 300, 4, 200, 98]
print(my_list)

print("Reverse a list")
# This can be done in two ways.

# Calling the reverse function
# This function will reverse the function in place.
# No new list will be returned
my_list.reverse()
print(my_list)

# Or calling by using broadcasting and indexes methods.
# This will return a new list but reversed
my_reversed_list = my_list[::-1]
print(my_reversed_list)

# Set it's used to avoid repeated values inside the list
print("Set of a list")
my_set_list = set(my_list)
print(my_set_list)

# Function to sort the list
print("Sort of a list")
my_list.sort()
print(my_list)

# Some other operations in Python.
my_list_1 = [0, 1, 2]
my_list_2 = [3, 4, 5]

# This is only limite to multiply with scalar
# Following is not allowef, unless you use another module such as Numpy
# my_mult_list = my_list_1 * my_list_2
print("Multiplication of arrays")
my_mult_list = my_list_1 * 3
print(my_mult_list)

print("Sum of arrays")
my_sum_list = my_list_1 + my_list_2
print(my_sum_list)


# All these functions are useful to manage simple expresion in python.
# However, for more functional or advanced arrays techniques and manipullation,
# it's used other modules like numpy or Panda, that manages vectorization a
# Pararell computing.


print("# SLICING")
my_str = "This is a sentence for terting."

# Slice from the initial
print(my_str[:10])
# Slice from the final
print(my_str[-10:-1])
# Slice from the middle
print(my_str[5:20])

# Also the slicing can be done in steps, increasing or decreasing
print(my_str[::2])
print(my_str[::-2])
# This is to show the string using one step only
print(my_str[::1])
print(my_str[::-1])

# You can find an element within the list by sung the in element
if "This" in my_str:
    print("The string has been founded")
else:
    print("Error: String not founded")

# You can use the split function to separate the array by using a character
items_splitted = my_str.split(" ")
print("Number of occurrences {0}".format(len(items_splitted)))
print(items_splitted)
# Get the second element found
print(items_splitted[2])

# In the same way you can append two arrays by using the "+"
# You can concatenate two strings using the same way
# However this is not working if you try tu sum a numerical value with a string
my_str_2 = "So this is already done."
print(my_str + " "+ my_str_2)

# You can use single quotes '' of double quotes "" to create strings in Python
print("This is a String")
print('This is another String')
