from os.path import join
from os import listdir, rmdir
from shutil import move
import os
import string

root = os.getcwd()
for Initial in list(string.ascii_lowercase):
    try:
        for filename in listdir(join(root, Initial)):
            move(join(root, Initial, filename), join(root, filename))
        rmdir(join(root, Initial))
    except:
        pass

##i = 0
##root = os.getcwd()
##for folder in listdir(root):
##    try:
##        for filename in listdir(join(root, folder)):
##            i += 1
##        if i != 1000:
##            print(folder)
##        else:
##            pass
##        i = 0
##    except:
##        pass
