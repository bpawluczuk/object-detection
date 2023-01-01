import numpy as np
from joblib.numpy_pickle_utils import xrange

tab = ["a", "b", "c", "d"]

print(tab[1:2])

print(tab[-2:])

arr = np.array(tab)

print(arr)
print(arr.reshape(1, -1))
print(arr.reshape(-1, 1))
print(arr.reshape(-1))

# Python code to demonstrate range() vs xrange()
# on basis of return type

# initializing a with range()
a = range(1, 10000)

# initializing a with xrange()
x = xrange(1, 10000)

# testing the type of a
print("The return type of range() is : ")
print(type(a))

# testing the type of x
print("The return type of xrange() is : ")
print(type(x))
