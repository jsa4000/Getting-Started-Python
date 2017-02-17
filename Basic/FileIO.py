# -*- coding: utf-8 -*-
"""
This class is an overview of how to work with very vasic file types
In this class we are going to view two files, plain/text and csv files


There are more file types to cover, however some modules and libraries
alrady provide this functionality.

Some of these files are:
- HDF5:
    DF5 is a data model, library, and file format for storing and managing data.
    It supports an unlimited variety of datatypes, and is designed for flexible
    and efficient I/O and for high volume and complex data. HDF5 is portable
    and is extensible, allowing applications to evolve in their use of HDF5.
    The HDF5 Technology suite includes tools and applications for managing,
    manipulating, viewing, and analyzing data in the HDF5 format.
- Pickle:
    The pickle module implements a fundamental, but powerful algorithm for
    serializing and de-serializing a Python object structure. “Pickling”
    is the process whereby a Python object hierarchy is converted into a
    byte stream, and “unpickling” is the inverse operation, whereby a byte
    stream is converted back into an object hierarchy. Pickling (and unpickling)
    is alternatively known as “serialization”, “marshalling,” [1] or “flattening”,
    however, to avoid confusion, the terms used here are “pickling” and “unpickling”.
 - Compressed files:
    Zip, rar, etc.. These files can be also compressed or uncompressed depending
    on the needs. It's not neccessary to uncompress the files to be able to Unpack
    the files and work with them.

"""
import csv
# pylint: disable=invalid-name

# NOTE: Following line has to be include if current file has non-ASCII enconding
#       I have copy paste some text from internet, so it has UTF-8 enconding.
# -*- coding: utf-8 -*-

#Text plain text
# Read/write

# The first thing to o is open the file.
#ncluding a mode argument is optional because a default value of ‘r’ will be assumed
#  if it is omitted. The ‘r’ value stands for read mode, which is just one of many.
#
#The modes are:
#
#    ‘r’ – Read mode which is used when the file is only being read
#    ‘w’ – Write mode which is used to edit and write new information to the file (any
#          existing files with the same name will be erased when this mode is activated)
#    ‘a’ – Appending mode, which is used to add new data to the end of the file; that
#          is new information is automatically amended to the end
#    ‘r+’ – Special read and write mode, which is used to handle both actions when
#           working with a file


print("Creating Text files")

# In order to separate the write data. It's neede to add \n\r to the
# end of each line, since there is no specific method to do it

# Open and Create text file (write mode).
#NOTE: "/" slashes are needed instead "\" for opening or creating files
file = open("C:/file.txt", "w")
# No we can start writing into the file by using the handle that open returns
file.write("Dear Yasmine,\n\r")
file.write("This is an example of how to open a file in write mode.\n\r")
# You can define an array to wirte multiple lines at the same time.abs
lines = ["Best Regards\n\r", "\n\r", "Javier\n\r"]
file.writelines(lines)

# finally it's needed to close the file after the operations
file.close()

print("Reading Text files")

# Open and read file
# We can start writing into the file by using the handle that open returns
# You can read the file:
#   1 - Reading a number of bytes file.read(5)
#   2 - Reading the entire file file.read()
#   3 - Reading the entire file but returning lines
#   4 - reading the file by using an interator (easy way)
#   5 - Reading the file by line of given the number of the line

# 1 - Reading a number of bytes file.read(5)
# Open text file (read mode)
file = open("c:/file.txt", "r")

# Lets start reading the first 5 bytes
readed = file.read(9)
print(type(readed))
print(readed)

# finally it's needed to close the file after the operations
file.close()

#2 - Reading the entire file file.read()
# Open text file (read mode)
file = open("c:/file.txt", "r")

# This function read the file entirely without line-bracks or anything.
readed = file.read()
print(type(readed))
print(readed)

# finally it's needed to close the file after the operations
file.close()

#   3 - Reading the entire file but returning lines
# Open text file (read mode)
file = open("c:/file.txt", "r")

# This function read the file entirely without line-bracks or anything.
readed = file.readlines()
print(type(readed))
print(readed)

# finally it's needed to close the file after the operations
file.close()

#   4 - reading the file by using an interator (easy way)
# Open text file (read mode)
file = open("c:/file.txt", "r")

# This function read the file entirely without line-bracks or anything.
for line in file:
    print(line)

# finally it's needed to close the file after the operations
file.close()

# Also you can read a line or a number of line
# Open text file (read mode)
file = open("c:/file.txt", "r")

# This function read the file entirely without line-bracks or anything.
# In the readline mthod suposely it's going to read the line specified index(x)
# parameters
readed = file.readline()
print(type(readed))
print(readed)

readed = file.readline(3)
print(type(readed))
print(readed)

# finally it's needed to close the file after the operations
file.close()


print("Reading csv files")

# Not let's take a look at csv files instead.
# We have the following csv file
"""
file.csv

id,name,phone,Job
0,Javier, 912333,Senior Software Engineer
1,Peter,9833333 ,Bus Driver
2,Sonia,234444,Recepcionist
4,Peter,932344354,Dancer
"""
# Not that thaere are some spaces after and bofore each columns. This is to
# know exactly the behaviour the readed will have to deal with theses values.

# Prior to be able to open and read csv files, it's needded to import csv module()

# Here we are going to use the "With" clausure.
# This mean don't be neccesary to close the file or dipose the elements,
# similar to other languages like Java or c#
# NOTE: Seems it isn't compatible with Python Version < 3 release
with open("c:/file.csv") as csvfile:
    #Now read the file as a list
    #In this case the openned file need to be passed to the Dictreader Function
    dbfile = list(csv.DictReader(csvfile))

#This is to get the first row from tyhe fetched rows
columnId = dbfile[0]
print(columnId)
# The output of this print it's the following
#{'id': '0', 'name': 'Javier', 'phone': ' 912333', 'job': 'Senior Software Engineer'}

#to get the entire dataset read you can use the len operation since it's a standard list.
print(type(dbfile))
print(len(dbfile))

# The returned values it's a list (Similar to dataframe in Pandas).
# Each item contains a dictionary, with the kays and the values that are availabile for all
# dictionaries. (This is similar to series in Pandas)

# To get the keys and the values,it can be used the following lines:
print(dbfile[0].keys())
print(dbfile[0].values())

# NOTE: the dictionary doesn't perserve the order. This is something you should be aware of.
# To get all the names you can use a list comprehension, just like follows:
names = list(dbrow["name"] for dbrow in dbfile)
print(names)

# in order to get the values but not repeated you can use an in-built function called set,
# similar to list, but thi will return a the list without repeated values
names = set(dbrow["name"] for dbrow in dbfile)
print(names)

# Also a sorten built-in function could be used to sorten the list
names = sorted(list(dbrow["name"] for dbrow in dbfile))
print(names)
# It can be used the inline function to modify directly the list itself
names = list(dbrow["name"] for dbrow in dbfile)
names.sort()
print(names)




