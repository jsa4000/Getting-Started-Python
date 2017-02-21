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

# Pandas DataFrames
print("DataFrames")

# In order to create a Data frame it's needed to create series.
# To sum up we we talk what a serie means and what data frame means

# Serie: it's a set of key and values.
# DataFrmae : it's a set of series with key and series.
#   - In a data frame the keys are the indexes of the series or the rows.
#   - In a data frame each key of the series will be the column name

# So let's we create a Dataframe

# Columns that will be added to the data frame
columns = ["id", "name", "description", "review"]

# Data for the series
serie1 = ["ID_1", "Gladiator", "Something related with Romans", "Very good film"]
serie2 = ["ID_2", "Shrek", "Fairy-tail with Cartoon characters", "Very funny film to look at"]
serie3 = ["ID_3", "Ghost in the Shell", "Japanese Anime movie", \
          "Old-shool animated film. A classic"]

print("Method #1: by passing all values and indexes")
# We can create a serie this way (Ordered)
ps1 = pd.Series(serie1, index=columns)
print(ps1.index)
print(ps1)

print("Method #2: by passing a dictionary with the keys (indexes) and values")
# We can create a serie this way (Non Ordered).
ps1 = pd.Series({columns[0]:serie1[0], columns[1]:serie1[1], columns[2]:serie1[2], columns[3]:serie1[3]})
print(ps1.index)
print(ps1)


print("Method #3: iterating by using a loop to create a dictionary")
# We can create a serie this way (Non Ordered).
lserie = {}
for i, value in enumerate(columns):
    lserie[columns[i]] = serie1[i]
    print(type(i))
    print(value)
print(lserie)
ps1 = pd.Series(lserie)
print(ps1.index)
print(ps1)

print("Method #4: List Comprehension")
# We can create a serie this way (Non Ordered). 
# List Comprehension
lserie={columns[i]:serie1[i] for i, value in enumerate(columns)}
print(lserie)
ps1 = pd.Series(lserie)
print(ps1.index)
print(ps1)

# Finally create the data frame
#the data frame will contain all the series and the indexes for the rows.
lserie1 = {columns[i]:serie1[i] for i, value in enumerate(columns)}
lserie2 = {columns[i]:serie2[i] for i, value in enumerate(columns)}
lserie3 = {columns[i]:serie3[i] for i, value in enumerate(columns)}
lseries = [pd.Series(lserie1), pd.Series(lserie2), pd.Series(lserie3)]
print(lseries)

sIndexes = [serie["id"] for serie in lseries]
print(sIndexes)

# Once the series and the indexes are created then create the dataframe
df = pd.DataFrame(lseries, index=sIndexes)
print(df)

# In this case the index has been repeated two times: 
#   - first columns serie 
#   - dataframe index.
# 
# The column with the index should be removed so it won't be inserted

# Let's constraint the list comprehesnion
lserie1 = {columns[i]:serie1[i] for i, value in enumerate(columns) if i > 0}
lserie2 = {columns[i]:serie2[i] for i, value in enumerate(columns) if i > 0}
lserie3 = {columns[i]:serie3[i] for i, value in enumerate(columns) if i > 0}
lseries = [pd.Series(lserie1), pd.Series(lserie2), pd.Series(lserie3)]
print(lseries) 

# Now let's create the index
sIndexes = [serie1[0], serie2[0], serie3[0]]
print(sIndexes)

# Once the series and the indexes are created then create the dataframe
df = pd.DataFrame(lseries, index=sIndexes)
print(df)

# Not let's wet a serie
print("PRINT THE SERIE WITHIN THE INDEX 0")
print(df.iloc[0])
print("PRINT THE SERIE WITH THE INDEX NAME ID_1")
print(df.loc["ID_1"])
print(type(df.loc["ID_1"]))

# Not if we try to write by using the normal index with the data frame.
# This is no longer allowed like in the series
# print(df["ID_1"]) # NOT ALLOWED
# print(df[0])       # NOT ALLOWED

# The indexing in Pandas if for fetching the columns instead.
# the iclo it's to fetch the desired rows first

print("Let's use indexing to acces to the columns")
print(df["name"])
print(df[columns[2]])

# To access to the serie we can finally use the methOd we want:
#    (index, loc, iloc, ix, etc, etc..)
print(df.loc["ID_1"]["name"])
print(df.loc["ID_1"][columns[2]])

# Instead, we can use 2D indexing isntead and request any range or mask
# We can also request more that one column in the indexing
print("Fancy indexing using PANDAS")
print(df.loc["ID_1", ["name", columns[2]]])
#We can also fetch all the rows or coulmns, instead filtering
print("Fancy indexing using PANDAS")
print(df.loc[:, ["name", columns[2]]])

# We can aslo applya a transpose to the data
# This will mean the colums will become the rows
# and the rows the columns
print("LET'S DO THE TRANSPOSE")
dfT = df.T
print(dfT)
print("Let's write the row named name and the column ID_1")
print(dfT.loc["name"]["ID_1"])

# Drop elements
print("Let's delete a row by using the index Id")
deldf = df.drop("ID_2")
print(df)
print(deldf)

# AS you can see the operation doesn't delete the content directly
# Instead a copy will be returned with the delete operation applied

# Yoy can also make a copy if your need it
print("Let's do a Copy")
my_copy = df.copy()
print(my_copy)

# Let's delete a column
# no you can delete a row if you want Python default operators
del my_copy["name"]
print(my_copy)

print("READ CSV file")
# Data frame Indexing and Loading
# For this particular example we are going to load the current csv
"""
file2.csv

id,product,description,cost,units
00001,HP Probook 6470b,Hardware component,1240.99,12
02111,SAMSUNG GALAXY,Tablet PC,340.59,120
00232,Xiaomi redmine 3,Mobile Phone,120,234
12330,HP Probook 6470b,Hardware component,1240.99,12
"""
df_csv = pd.read_csv("C:/file2.csv", sep=",")
# REturns the first 5 elements with head
print("*****HEAD")
print(df_csv.head())
# REturns the last 5 elements with head
print("*****TAIL")
print(df_csv.tail())
# print all the elements 
print(df_csv)

# Let get the product name and it's cost into a view
print(df_csv["product"])
# We can give a list with the columns to fetch
print(df_csv[["product", "cost"]])
# Following will get an exception. In order to get a concrete row you need to use iloc or loc
#print(df_cvs[0, ["product","cost"]]) # ERROR
print(df_csv.loc[0, ["product", "cost"]])
# We can aslo retrieve all the files like normal indexing using numpy indexing
print(df_csv.loc[:, ["product", "cost"]])
# print(df_cvs.loc[:, :2]) # This is not allowed
print("Return the first two rows")
print(df_csv[:2])

print("Let's oeprate with the data")
cost = df_csv["cost"]
print(cost)
# We could sum 2 for all the elements the view
cost+=2
print(cost)

print("OPTIONS READING CSV file")

# This will allow some parameters reading the csv file
# - Folloing will skip the header, so the header will be read
#   from row 1. And 
# - Also we can specify the column to choose for the index
df_csv = pd.read_csv("C:/file2.csv", sep=",", index_col=1, skiprows=1)
print(df_csv.head())

# Let's ope normally the file
df_csv = pd.read_csv("C:/file2.csv", sep=",")
#Following will return a list with all the columns in the dataframe
print(df_csv.columns)

#Raname a columns. 
# 
# This will introduce inplace operations that will be applied directly into
# the dataframe isntead return a new df with the changes.
df_csv.rename(columns={"product":"new_col"}, inplace=True)
print(df_csv.columns)

#Querying a Dataframe
# This is similar to Numpy. So you can create masks in order to get a view
import numpy
my_array = numpy.linspace(0,12,12)
my_fancy_array = my_array[my_array>6]
#This will create an array with the mask
print(my_array>6)
# finall let's apply the mask to the numpy array
print(my_fancy_array)
# Now let's look how it works using Numpy
# Lst's get the cost higher that 600
my_filter_df = df_csv["cost"] > 600
# Similar to the numpy behaviour this will create a mask with the filter
print(my_filter_df)
#Finally apply the mask to the dataframe
print("Using indexing for the Query")
print(df_csv[my_filter_df])
print(df_csv[my_filter_df].count())
#Similar we can use the whee function implemented.
print("Using where function for the Query")
print(df_csv.where(my_filter_df))
print(df_csv.where(my_filter_df).count())
new_filter = df_csv.where(my_filter_df)

#Both methods returns the same correct, values. However where function
# retunr null values for the rows that don't satisfy the condition.

#in order to clean the omepy values using the where cmethod you can
# use the dropna functio
print("Drop NaN values from the query")
new_filter.dropna()
print(new_filter)

# No inline so
removed_nan = new_filter.dropna()
print(removed_nan)

print("USING MORE THAT ONE CONDITION")

# You can also use more than one filter
#It's important toput parenthesis between the values to check

# NO RESULTS 
#my_filter_df = df_csv["cost"] >= 100 & df_csv["cost"] <= 600

# GOOD RESULTS
my_filter_df = (df_csv["cost"] >= 100) & (df_csv["cost"] <= 600)
print(my_filter_df)
print(df_csv[my_filter_df])

# Something that I have confused a bit it's the way to access to
# the elements and the best option to use.
# Indexing directly with [] have diferent behaviours sime times
print(df_csv)

#if you want to get the first two rows you can use
print(df_csv[:2])
# Also if you want to get a row, eaxmple 0
# print(df_csv[0]) # ERROR !!!!!!!
#In this case you have to spedify the column with all the rows
print(df_csv["id"])

#if you can to get a row you need to use loc
print(df_csv.loc[0])
# Also you can specify the row and columns to fetch
print(df_csv.loc[0, ["id", "cost"]])
print(df_csv.loc[:, ["id", "cost"]]) # Get all the values ":"

# Indexing a data Frame.
print("Indexing a Data Frame")

# you can add a new column using indexing like normal dictionaries
df_csv["Date"] = "None"
print(df_csv.head())

#if you want to chage your index it's recommended first store your current
# index into a new column
#As this use numpy the values will be modified element-wise since it's a 
# matrix sum, from the elements from one matrix and the vector with the indexes.
df_csv["id_index"] = df_csv.index
print(df_csv.index)
print(df_csv.head())

#To set the new index you need to use set_index followed by the column
df_csv.set_index("id")
print(df_csv.index)
print(df_csv.head())

# Remember that usually dataframes will return a copy so the original it's not
# going to be modified.
df_csv = df_csv.set_index("id")
print(df_csv.index)
print(df_csv.head())

# You can also set multiple indexed
df_csv = pd.read_csv("C:/file2.csv", sep=",")
print("Current CSV loaded")
print(df_csv.head())
print("CSV with new indexes")
df_csv = df_csv.set_index(["id","product"])
#print(df_csv.index)
print(df_csv.head())

# if you want to acces to an element you can use
# returns the id and the column yo want
print(df_csv["cost"].head())
# You can get a give item by knowing the key 
print(df_csv.loc[1,"HP Probook 6470b"])
# Or an array with multiple elements (tuple)
print(df_csv.loc[[(1,"HP Probook 6470b"),(12111,"SAMSUNG GALAXY")]])


#Similar to set you can get the unique items sepecifinf the column from the series
#you want to use.
unique_elems = df_csv["cost"].unique()

#You can get a new view with the columns you want ()in a list
new_view = df_csv[["cost","description"]]
print(new_view.head())

#Reset the index at it's initial stage with the internal id
# Also all the columns previously indexes will be added as 
# default
print(df_csv.head())
df_csv = df_csv.reset_index()
print(df_csv.head())

# Let change the index again
df_csv = df_csv.set_index(["product","units"])
print(df_csv.head())
# Sort the index
df_csv = df_csv.sort_index()
print(df_csv.head())

#Clean Missing Values
print("Clean Missing Values")

# In order to fill the missing values there is an automatic method
# that allow you this functionality
df = df.fillna(method='ffill')
df.head()






