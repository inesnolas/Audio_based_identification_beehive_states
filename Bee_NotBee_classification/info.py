import numpy as np
import sys

### For prints with timetag in front ###

from datetime import datetime
def timestr():
    return '[' + datetime.now().strftime('%Y-%m-%dT%H:%M:%S') + ']'

import os
import psutil
def timeRAMstr():
    pid = os.getpid()
    py = psutil.Process(pid)
    return '[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' RAM' + str(psutil.virtual_memory().percent) + '% ' + str( round(py.memory_info().rss/1024/1024/1024, 2)) + 'GB]'

def my_decorator(func):
    def wrapped_func(*args,**kwargs):
        return func(timeRAMstr(),*args,**kwargs)
    return wrapped_func

print = my_decorator(print)


### For colors ###

if sys.platform == 'win32' or sys.platform == 'win64':
    import colorama # makes colors work in windows terminal
    colorama.init()

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def printb(toprint):
    print(color.BOLD + toprint + color.END)

def printr(toprint):
    print(color.RED + toprint + color.END)

def printp(toprint):
    print(color.PURPLE + toprint + color.END)


### Info function ###

def i(variable):
    if type(variable) is np.ndarray and variable.size > 1:
        element = variable[0]
        while type(element) is np.ndarray:
            element = element[0]
        toprintelement = str(type(element))
        #toprintarray = "   " + str(variable.size)
    else:
        toprintelement = "N/A"

    #if type(variable) is dict:
        #toprintdict = str(len(variable))
    
    toprintstr = str(type(variable))
    if type(variable) is tuple:
        toprintstr = toprintstr + "   " + str(variable)
    try:
        toprintstr = toprintstr + "   size:" + str(variable.size)
    except (AttributeError):
        toprintstr = toprintstr + "   N/A" 
    try:
        toprintstr = toprintstr + "   shape:" + str(variable.shape)
    except (AttributeError):
        toprintstr = toprintstr + "   N/A" 
    toprintstr = toprintstr + "   " + toprintelement 
    try:
        toprintstr = toprintstr + "   len:" + str(len(variable))
    except (TypeError):
        toprintstr = toprintstr + "   N/A" 
    try:
        toprintstr = toprintstr + "   min:" + str(variable.min())
    except (AttributeError):
        toprintstr = toprintstr + "   N/A" 
    try:
        toprintstr = toprintstr + "   max:" + str(variable.max())
    except (AttributeError):
        toprintstr = toprintstr + "   N/A" 
    print(toprintstr)
    #return toprintstr
    
# Test function
#rab=2
#i(rab)

##def happyBirthdayEmily():
    #print("Happy Birthday to you!")
    #print("Happy Birthday to you!")
    #print("Happy Birthday, dear Emily.")
    #print("Happy Birthday to you!")

#happyBirthdayEmily()
