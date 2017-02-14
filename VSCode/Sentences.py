"""
This is a document to explain how the sentneces works in python
Note that this will be a continuation for the Sequence Basic examples
"""
# pylint: disable=invalid-name

# Something that you can do in python is to multi-assign variables
# Sometims this is very useful to return in your functions
# This is to return multiple values inside a function
my_array = [0, "Javier", "912343434"]
id_array, name, phone = my_array
print("{0}, {1}, {2}".format(id_array, name, phone))

def get_info():
    """
    This function will return three values after the function.
    """
    return  [0, "Javier", "912343434"]

id_array, name, phone = get_info()
print("{0}, {1}, {2}".format(id_array, name, phone))

# In dictionaies you can iterate over the elements by using the items, keys or the values
my_dict = {"0":"Car", "1":"Plane", "2":"Bus", "3":"Train"}

# By index
for item in my_dict:
    print(item + ":"+ my_dict[item])

#By keys
for key in my_dict.keys():
    print(key + ":"+ my_dict[key])

#By values
for value in my_dict.values():
    print(value)

# By mix of values
# For version of Python under 3.0 use instead iteritems()
for key, value in my_dict.items():
    print(key + ":" + value)

print("# APPEND vs EXTEND")
# Append vs extend. Both are functions that directly apply the op.
my_list = [0, 1, 2, 3, 4]
my_list2 = [5, 6]
print("Append function")
my_list.append(my_list2)
print(my_list)
print("Append function")
my_list = [0, 1, 2, 3, 4]
my_list.extend(my_list2)
print(my_list)

# As you can see extend is like addin the items individually into the list
# Append is like inseting a new elements. In this case is an array.

print ("# List Comprehensions")
# This is an advanced feature
# This is used to shorten the sencentes to iterate between elements.
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# In this example is to get the same array but powered by two n**2
# the normal way is the following
new_array = []
for value in my_list:
    new_array.append(value**2)
print(new_array)

# However there is a cool method to do the same
new_array = [value**2 for value in my_list]
print(new_array)
print(type(new_array))

# Also you can contrain some values inside the list comprehension if / else
new_array = []
for value in my_list:
    if value > 5:
        new_array.append(value**2)
    else:
        new_array.append(value)
print(new_array)
# List Comprehension Method
new_array = [value**2 if value > 5 else value for value in my_list]
print(new_array)
# Tuple Comprehension Method :)
new_array = (value**2 if value > 5 else value for value in my_list)
print(new_array)

# You can append or concatenate multiple list comprehension into one.
my_list1 = [0, 1, 2]
my_list2 = [3, 4, 5]
my_list3 = [6, 7, 8]
my_lists = [my_list1, my_list2, my_list3]
# The idea is to join all the values into one.
my_list1.extend(my_list2)
my_list1.extend(my_list3)
print(my_list1)
# Create again the list since the could be altered
my_list1 = [0, 1, 2]
my_list2 = [3, 4, 5]
my_list3 = [6, 7, 8]
my_lists = [my_list1, my_list2, my_list3]
# By using loops
new_array = []
for curr_list in my_lists:
    for value in curr_list:
        new_array.append(value)
print(new_array)
# See the way the for are placed in the list comprehesion compared
# to the way they are placed in the for the nested for
new_array = [value for curr_list in my_lists for value in curr_list]
print(new_array)

print("# Map function")

#This function is to apply a functions to a collections of items
def power(number, exp=2):
    """
    This function returns the power of a number.
    By default the exp will be 2. This is the square of the number.
    """
    return number ** exp

print("The square of 3 is {}".format(power(3)))

#So continuing with the same example of the previous one.
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
my_map_list = map(power, my_list)
print(my_map_list)
print(type(my_map_list))
# As you can see th returned values of the map function is not an array
# there are two ways for geting the values of a map function
new_list = []
for value in my_map_list:
    new_list.append(value)
print(new_list)
#Or directly casint the object that is returned by the function to a list
my_map_list = map(lambda x: x**2, my_list)
new_list = list(my_map_list)
print(new_list)
#Note that once you have loop over the map you cannot get the list using the casting

















