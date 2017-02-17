"""
This is the Pandas basic guide for Python

pandas is a software library written for the Python programming language
for data manipulation and analysis. In particular, it offers data structures
and operations for manipulating numerical tables and time series. pandas
is free software.abs

The name is derived from the term "Panel data", an econometrics term
for multidimensional structured data sets.

"""
import pandas as pd
import numpy as np
# pylint: disable=invalid-name

print("Pandas information")

# To get the available methods/attributes of Pandas module use dir
#dir(pd)

# To get the available help of Pandas module use dir
#help(pd)
#help(pd.Series)

print("Create Series: Only data")

# Create a series from Strings (dtype: object)
my_serie = ["House", "Computer", "Labtop", "Fridge"]
my_pd_serie = pd.Series(my_serie)
print(my_pd_serie)

# Create a series from Numbers (dtype: int64)
my_serie = [1, 2, 3, 4]
my_pd_serie = pd.Series(my_serie)
print(my_pd_serie)

# Create a Mix of series
my_serie = ["House", "Computer", 1, None]
my_pd_serie = pd.Series(my_serie)
print(my_pd_serie)

print("Check NaN values")

#In Numpy NaN it's an undefined value (None). In Panda it's the same
my_serie = [1, 2.3, 1, None]
my_pd_serie = pd.Series(my_serie)
print(my_pd_serie)
print(my_pd_serie[3])
print(np.NaN)

if np.NaN == my_pd_serie[3]:
    print("Both NaN are the same in pandas and numpy")
elif np.isnan(my_pd_serie[3]):
    print("Checking is a value is NaN is the same in pandas and numpy")

print("Create Series: Index + Data")

# Now lets create series from a dictionary
myseries = {"1":"house", "2":"Computer", "3":"Labtop", "4":"Fridge"}
my_pd_serie = pd.Series(myseries)
print(my_pd_serie)
print(my_pd_serie.index)

# Now let's create a serie directly from the functio
my_pd_serie = pd.Series(["House", "Computer", "Labtop", "Fridge"], index=["i1", "i2", "i3", "i4"])
print(my_pd_serie)
print(my_pd_serie.index)

print("Query Series")

#To access to the Pandas series you can use the same indexing operations
# So you can use the key 
print(my_pd_serie["i1"])
print(my_pd_serie["i4"])
# Or the position)
print(my_pd_serie[0])
print(my_pd_serie[3])

# This only will works if your key it's an string.

my_pd_serie_int = pd.Series(["House", "Computer", "Labtop", "Fridge"], index=[1, 2, 3, 4])
# Yo need to sue the correc tkey
print(my_pd_serie_int[1])
# I f you use the position since you have redefined the key it will give you an error
#print(my_pd_serie_int[0])

# However there are two other methods you can use iloc and loc
# Those tow methods will return a pandas object
#   WARNING: USE square brackets: [0] not (0) !!!

# loc, iloc vs ix, i

#loc is label based indexing so basically looking up a value
#  in a row, iloc is integer row based indexing, ix is a general
#  method that first performs label based, if that fails then it
#  falls to integer based. at is deprecated and it's advised you
#  don't use that anymore. The other thing to consider is what you
#  are trying to do as some of these methods allow slicing, and column
#  assignment, to be honest the docs are pretty clear:
#            pandas.pydata.org/pandas-docs/stable/indexing.html


# Locs is to access from the current position in the dict
print(my_pd_serie.iloc[0])
# Also you can use loc to access using the index
print(my_pd_serie.loc["i1"])


