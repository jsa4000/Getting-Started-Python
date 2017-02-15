import cPickle
import os

#Create a dictionary with the data (objet ot serialize)
dict = {"pedro":21, "jaime": 32, "almudena": 65}
print (dict["pedro"])

#############
# DUMP DATA #
#############

# Open the file in write and binary mode
file1 = open("test.pkl","wb")
# Dump the object into the file
cPickle.dump(dict,file1)
#Finally close the file
file1.close()

################
# RESTORE DATA #
################

# Open the file with the dump
file2 = open("test.pkl","rb")
# Load the file and save into a variable
dict2 = cPickle.load(file2)
# Close the file
file2.close()

#Print the data restored
print (dict2)
