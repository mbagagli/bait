"""
This module contains custom error message for BAIT picking algorithm

REFERENCE:
https://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python
"""


class Error(Exception):
    """ Base class for other exceptions """
    pass


class BadInstance(Error):
    """ Raised when important instance checks are not respected """
    pass


class BadKeyValue(Error):
    """ Raised when important instance checks are not respected """
    pass


class MissingVariable(Error):
    """ Raised when a parameter/value is missing """
    pass


class MissingKey(Error):
    """ Raised when a parameter/value is missing """
    pass


class MissingAttribute(Error):
    """ Raised when an attribute is not found """
    pass


class SizeMismatch(Error):
    """ Raised when incongruences between vectors happen """
    pass


class InvalidParameter(Error):
    """ Raised when an invalid parameter is passed to function/method
    """
    pass


class InvalidAttribute(Error):
    """ Raised when an invalid attribute is selected from a class """
    pass


class Miscellanea(Error):
    """ Raised when the developer is too lazy to think at smth else """
    pass
