
massage1 = "Learn Python. You'll need it."
print(massage1)
massage2 = 'You better learn it sooner than later.'
print(massage2)
# why two ways to make a string
#massage3 = 'Learn Python. You'll need it.' # error
massage3 = 'Learn Python. You\'ll need it.' # scape character (ec) avoids error
print(massage3)
massage4 = 'He said: "I\'m not sure"' # here we cannot escape from using ec
print(massage4)
# a way that will work always
print('----------------')
massage5 = '''A text "double coted"
and
'single coted'
and with multiple lines''' # one could use also """xxx"""
print(massage5)
