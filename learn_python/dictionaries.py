# dicts are use to store pairs input:output of data
# input = key and output = value

# ff post
'''
user_id = 209
massage = "Hello"
language = "English"
datetime = "20191028110500"
location = (44.59,-104.71)
post = {"user_id":209,"massage":"Hello","language":"English",
        "datetime":"20191028110500","location":(44.59,-104.71)}
print(post)
print(type(post))
'''

# post 2 (another way to construct a dictionary)
post2 = dict(massage="Hello",language="English")
print(post2)
# add more data
post2["user_id"] = 208
post2["datetime"] = "20191028111300"
print(post2)
# access to elements
print(post2['massage'])
# way to avoid key errors (and continue to execute code)
try:
    print(post2['location'])
except:
    print("post2 does not have a location")


# other functions
print(dir(post2))
loc = post2.get('location',None)
print(loc)

# iterate over the keys in a dictionary
for key in post2.keys():
    value = post2[key]
    print(key,'=',value)

for key, value in post2.items():
    print(key,'=',value)
