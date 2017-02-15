def first(x):
    return x + 10

def second(x):
    return x + 100

def third(x):
    return x + 1000

x = 0
y = first(second(third(x)))
y = lambda x: first(second(third(x)))
mylist = [first,second,third]

y = first
for item in mylist[1:]:
    y = item(y)
        
y = mylist[-1]
for item in mylist[::-1]:
    y = item(y)

list2 = [0,1,2]
inverseList = list2[::-1]
for item in inverseList[1:]:
    print(item)

inverseList = mylist[::-1]
for item in inverseList[1:]:
    print(item)


mylist = [0,1,2,3]

print (functions[1:])

def generalfunction (functions):
    if (len(functions)):
        function = generalfunction(functions[1:])
        print (function)
        return function

print generalfunction(mylist)


def first(x):
    return x + 10

def second(x):
    return x + 100

def third(x):
    return x + 1000

mylist = [first,second,third]

#def forward_prop (x):
#    y = mylist[0]
#    for forward in mylist[1:]:
#        y = forward(y)
#    return y

def forward_prop (x):
     y = x
     for forward in mylist:
         y = forward(y)
     return y

x= 3
y = third(second(first(x)))
y2 = forward_prop(x)

if (y == y2):
    print ("Ok")
else:
    print ("KEEP WORKING MAN")