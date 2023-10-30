import datetime
'''
print(dir(datetime)) # dir(x) mostra o que há em x
gvr = datetime.date(1956,1,31)
print('date = ',gvr)
mill = datetime.date(2000,1,1)
dt = datetime.timedelta(100) # para subtrair dias, coloca negativo
print('mill+dt = ',mill+dt)
'''

# default format: yyyy-mm-dd (there are several codes for other formats)
'''
launch_date = datetime.date(2017,3,30)
launch_time = datetime.time(22,27,0)
launch_datetime = datetime.datetime(2017,3,30,22,27,0)
print(launch_date)
print(launch_time)
print(launch_datetime)
print(launch_time.hour)
print(launch_time.minute)
print(launch_time.second)
print(launch_date.year)
print(launch_datetime.month)
'''

# current datetime
'''
t1 = datetime.datetime.today()
print('t1 = ',t1)
#print(now.microsecond) # time is given with microsecond resolution
for j in range(0,10**8):
    j = j**2
t2 = datetime.datetime.today()
print('t2 = ',t2)
dt = t2-t1
print('dt = ',dt)
'''

# string to datetime convetion
moon_landing = "7/20/1969"
moon_landing_datetime = datetime.datetime.strptime(moon_landing,"%m/%d/%Y")
print(moon_landing_datetime)
print(type(moon_landing_datetime))
