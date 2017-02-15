
"""
Class Car
"""

class Car(object):
    """
    Doc for the class
    """
    def __init__(self, name):
        """
        Initialize the memeber of the current class
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
        This is to write-mode property. This is used to set the value of a property
        """
        self._wheels = number

