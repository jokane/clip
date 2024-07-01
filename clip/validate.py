""" Methods to validate data types. """

def is_float(x):
    """ Can the given value be interpreted as a float? """
    try:
        float(x)
        return True
    except TypeError:
        return False
    except ValueError:
        return False

def is_int(x):
    """ Can the given value be interpreted as an int? """
    return isinstance(x, int)

def is_string(x):
    """ Is the given value actually a string? """
    return isinstance(x, str)

def is_positive(x):
    """ Can the given value be interpreted as a positive number? """
    return x>0

def is_even(x):
    """ Is it an even number? """
    return x%2 == 0

def is_non_negative(x):
    """ Can the given value be interpreted as a non-negative number? """
    return x>=0

def is_color(color):
    """ Is this a color, in RGB 8-bit format? """
    try:
        if len(color) != 3: return False
    except TypeError:
        return False
    if not is_int(color[0]): return False
    if not is_int(color[1]): return False
    if not is_int(color[2]): return False
    if color[0] < 0 or color[0] > 255: return False
    if color[1] < 0 or color[1] > 255: return False
    if color[2] < 0 or color[2] > 255: return False
    return True

def is_int_point(pt):
    """ Is this a 2d point with integer coordinates. """
    if len(pt) != 2: return False
    if not is_int(pt[0]): return False
    if not is_int(pt[1]): return False
    return True

def is_iterable(x):
    """ Is this a thing that can be iterated? """
    try:
        iter(x)
        return True
    except TypeError:
        return False

def require(x, func, condition, name, exception_class):
    """ Make sure func(x) returns a true value, and complain if not."""
    if not func(x):
        raise exception_class(f'Expected {name} to be a {condition}, '
                              f'but got a {type(x)} with value {x} instead.')

def require_iterable(x, name):
    """ Raise an informative exception if x is not iterable. """
    require(x, is_iterable, "iterable", name, TypeError)

def require_int(x, name):
    """ Raise an informative exception if x is not an integer. """
    require(x, is_int, "integer", name, TypeError)

def require_float(x, name):
    """ Raise an informative exception if x is not a float. """
    require(x, is_float, "float", name, TypeError)

def require_string(x, name):
    """ Raise an informative exception if x is not a string. """
    require(x, is_string, "string", name, TypeError)

def require_color(x, name):
    """ Raise an informative exception if x is not a color. """
    require(x, is_color, "color", name, TypeError)

def require_int_point(x, name):
    """ Raise an informative exception if x is not a integer point. """
    require(x, is_int_point, "point with integer coordinates", name, TypeError)

def require_positive(x, name):
    """ Raise an informative exception if x is not positive. """
    require(x, is_positive, "positive number", name, ValueError)

def require_even(x, name):
    """ Raise an informative exception if x is not even. """
    require(x, is_even, "even", name, ValueError)

def require_non_negative(x, name):
    """ Raise an informative exception if x is not 0 or positive. """
    require(x, is_non_negative, "non-negative", name, ValueError)

def require_equal(x, y, name):
    """ Raise an informative exception if x and y are not equal. """
    if x != y:
        raise ValueError(f'Expected {name} to be equal, but they are not.  {x} != {y}')

def require_less_equal(x, y, name1, name2):
    """ Raise an informative exception if x is not less than or equal to y. """
    if x > y:
        raise ValueError(f'Expected "{name1}" to be less than or equal to "{name2}",'
          f' but it is not. {x} > {y}')

def require_less(x, y, name1, name2):
    """ Raise an informative exception if x is greater than y. """
    if x >= y:
        raise ValueError(f'Expected "{name1}" to be less than "{name2}", '
          f'but it is not. {x} >= {y}')

def require_callable(x, name):
    """ Raise an informative exception if x is not callable. """
    require(x, callable, "callable", name, TypeError)

