"""
This file is going to explain some of the functionailty of the Classes
"""
# pylint: disable=invalid-name

#exec
print("#Creating Classes ")

# Following a basic definition of a Class using Python

class Car(object):
    """
    Documentation for the class
    """

    # Global variables shared for all the instances of Car
    class_name = "Car"

    def __init__(self, name):
        """
        Initialize the members of the current class
        """
        self.__name = name
        self._wheels = 4

    @property
    def name(self):
        """
        Property called name. Doesn't have setter, that means only-read mode.
        __ : for private member: RECOMMENDED NOT TO USE
        _ : for private-protected memebers
        """
        return self.__name

    def _initialize(self):
        """
        Private function. 'pass' it's used when the function doesn't do anything
        """
        pass

    @property
    def wheels(self):
        """
        Property to get the number of wheels current car has
        """
        return self._wheels

    @wheels.setter
    def wheels(self, number):
        """
        Set the value of a property
        This is to set write-mode to the Property.
        """
        self._wheels = number

#Let's test the class

# We can acces to the static variable with no instance created yet
print(Car.class_name)

#Create the class with the default values
car = Car("MyCar")

print(car.class_name)
print(car.wheels)
print(car.name)

# Following call will give us a warning since it's a private member
print(car._wheels)

# Modify private members it's allowed but it's not recommended.
# We could create a setter propery instead
car.__name = "Private Memeber being Modified"

# Cannot set name attribute directly, since there is no setter created for the property
#car.name = "Member being Modified"

# Modify the wheels by useing the setter propery
car.wheels = 16

# Modify statics variable of Car
car.class_name = "Modified_Car"

# Print current values 
print(car._wheels)
print(car.wheels)
print(car.name)

#not let's create another instance of car
print(Car.class_name)
car2 = Car("MyCar2")

print(car2.wheels)
print(car2.name)

#Let's set the static variable
car2.class_name = "Modified_Car_2"

print(Car.class_name)
print(car.class_name)
print(car2.class_name)

#No modificartions..
Car.class_name = "Reset_static_Car"

print(Car.class_name)
print(car.class_name)
print(car2.class_name)


#exec
print("# *args and **kwargs")
# *args and **kwargs

# First at all let's see how the args are passed into the functions
#  - First the args need to be listed in the same order than defined
#    originally in the Function
#  - Sencondly the named arguments. The arguments can be alter the
#    order of them, since some attributes can be null or optional 

def myFunction(name, address, age=0, city=None, dead=False):
    """
    Some example of the function
    """
    # Do something
    pass

# Following examples are good
myFunction("Javier","C\GranVia")
myFunction(name = "Javier",address="C\GranVia")
myFunction(address="C\GranVia", name = "Javier")
myFunction("Javier","C\GranVia",22)
myFunction("Javier","C\GranVia",dead=True)
myFunction("Javier","C\GranVia",34,True)

#Bad calls [COMMENTED]

# Mandatory parameters needed
#myFunction()
#myFunction("Javier")
#myFunction("C\GranVia")
#myFunction(city=None)
#myFunction("Javier","C\GranVia",True, 33) # Wrong order

# Wrong order of parameters
#myFunction(city="Madrid","Javier","C\GranVia")

# args or unnamed arguments.

def test_args(f_arg, *args):
    """
    This is a test showing how args works
    """
    print("Normal argument: {}".format(f_arg))
    for arg in args:
        print("Argument through *argv : {}".format(arg))

# Following to test the function and the evaluation of args.
#   - The first arg is used as a normal args since the function allow it
#   - Following are the args that will be added into a list of arguments,
#      since in the function are not inlcuded.
# The args can be useful to ingnore some of the attrbiute or to clean a little
# bit the code

test_args('yasoob','python','eggs','test')

#kwargs are the named parameters that are used in the call
def test_kwargs(**kwargs):
    """
    This function will return all the named values within the call
    """
    if kwargs is not None:
        for key in kwargs:
            print(("{0} == {1}").format(key, kwargs[key]))

test_kwargs(name="yasoob")

# Following a function that could be used inside a function to get all the args
# and kwargs into a list.
def extract_args_kwargs(locals):
    """
    Function that returns a list with the args and kwargs by using local() 
    funtion in Python
    """
    params = {}
    parameters = locals.keys()
    parameters.remove("self")
    for key in parameters:
        if key == "kwargs":
            params.update(locals.get(key))
        else:
            params[key] = locals.get(key)
    return params


print("# Inheritance")
# Inheritance.

# Following example could be very useful to serializa or deserialize an object.
# For example:
#
#       json = {"Class": "BaseClass","name":"MyName", "params":"Myparams"}
# 
#  You can create a Base class object by
#   
#      myJSONObject = BaseClass(json)
#
#  Then you will have access to all the parametes and attributes.

#
# If your have a child class but the base class it's Base class then.
#
#     json = {"Class": "ChildClass","name":"MyName", "params":"Myparams"}
#      myJSONObject = BaseClass(json)
#
# Finally cast the object from The base Class to the ChildClass by simply
#  NOTE:  The class must be serialized or evaluated so Python can recognize
#           it as a class type
#     myJSONObject.__class__ = myJSONObject._Class


class BaseClass(object):
    """
    This is the main class (base) we generate. This will inherit from object
    In this Class the constructor will allow to recieve any parameter
    For eaach paramters, an attribute will be created so will be part
    of the Class. The memebers creates will be private "_{}".format(key)
    """

    def __init__(self, **kwargs):
        """
        Constructor
        """
        self._attrs = {}
        for key in kwargs:
            # Add the attribute to the list of attributes
            self._attrs[key] = kwargs[key]
            # Create an attribute s we can access directly typing the member.
            # e.j. some._name
            setattr(self, "_{}".format(key), self._attrs[key])

    @property
    def attrs(self):
        """
        Property to access to all the attributes of the class
        """
        # following, we will return all the attributes with the current values.
        # This will update the current attrs list
        for key in self._attrs.keys():
            self._attrs[key] = getattr(self, "_{}".format(key))
        return self._attrs

    def set_attr(self, key, item):
        """
        Public method that allows to set new attribute to the current Object
        If it already exists then the attribute will be updated with the current
        value or item.
        """
        self._attrs[key] = item
        setattr(self, "_{}".format(key), self._attrs[key])

    def del_attr(self, key):
        """
        Public method that delete the attribute from the list.
        Also the attribite will be deleted from the class so it cannot be
        accesses anymore.
        """
        del self._attrs[key]
        delattr(self, "_{}".format(key))

class BaseNode(BaseClass):
    """
    This class will inherit from Base Class.
    So all the functionality will be also used from this child class.
    """
    def __init__(self, name=None, shape=None, **kwargs):
        """
        Construcutor of the class
        """
        # Parse all the parameters of the function (We could use locals())
        # We can do also the following:
        #       params[kwargs.keys[0]] = kwargs[kwargs.keys[0]])
        params = extract_args_kwargs(locals())
        # We could create additional params used for this class
        params["params"] = []
        # Finally call to the Base class to initialize
        super(BaseNode, self).__init__(**params)

    def __call__(self, name=None):
        """
       __call__ function doesn't return a new object nor inititialize it.
       The objective if that it can by invoked like a function to do something.
       For example:
            - To change its properties by calling directly like a constructor or function.
            - To do something else, like return some other value or compute anything.
       e.j.
            object = Object("name")
            object("New_name")

            # Another way could have been the following
            object._name = "New_name"

        """
        super.set_attr("name", name)
        return self


print("# internal variables in Python")
# Members and internal variables in Python
# __class__, __type__, __call__, __init__, etc..

# Following are the explanation about how __call__ works.

class Test(object):
    """
    Tast class
    """

    def __init__(self, name):
        self._name = name
        self.age = 10

    def __call__(self, name):
        self._name = name
        return self


#Create the initial objects
mytest = Test("Javier")
mytest2 = Test("Pablo")
mytest2.age = 12

print(mytest._name)
print(mytest.age)

print(mytest2._name)
print(mytest2.age)

#We use the __call__ function implemented in the class
mytestCall = mytest("JavierSantos")

# The objet ifselt has been modified
print(mytest._name)
print(mytest.age)

#The returned object it's current modified
print(mytestCall._name)
print(mytestCall.age)

# This object remains the same
print(mytest2._name)
print(mytest2.age)

# These are some function iimplemented in Python class

#To se how to get the current class of an instance use __class__
print(mytest2.__class__)
# In order to get the name (str) of a Class type
print(Test.__name__)

#To check if a instance is of a specific type.
if isinstance(mytest2, Test):
    print("OK")

if isinstance(mytest2, list):
    print("OK")
else:
    print("NO")

#You can know if a class it's a subclass of another by following command
class Test2(Test):
    """
    Subclass
    """
    pass

if issubclass(Test2, Test):
    print("OK")

#Since Python is typeless the is no need to cast any object into another.
# However some times you need to cast your object so you can access to
# some of the functionality of that object.
# An example of this could be to deserializae an object from eval and them
# convert it into an object.

# In order to do this you can modifcay the parameter __class__ of your data.

def cast(data, type):
    """
    Function to cast the date passes by parameters into a type
    """
    data.__class__ = type
    return data

