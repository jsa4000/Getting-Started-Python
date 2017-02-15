
theano = False

def function(f):
    def wrap(*args, **kw):
        if (theano):
            return f(*args, **kw)
        else:
            return f
    return wrap


@function
def multiply (x,y):
    return x * y

@function
def square(x):
    return x**2

if (theano):
    x = multiply (4,2)
    print (x)

    y = square (3)
    print (y)
else:
    x = multiply (4,2)
    print (x) 
    print (x(4,2)) 
    y = square (3)
    print (y)
    print (y(3))


