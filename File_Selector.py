import os
import pandas as pd
from pathlib import Path
from collections import Counter
import shutil

path = Path('C:/Users/Jokin Lasa Escobales/Desktop/Data/Master/Thesis/MERRA2/MERRA2')
files = [file.stem for file in path.rglob('*')]
files2d = []
files3d = []
v = 0
print('files')
print(files)

# separating 2d and 3d files
while v < len(files):
    if '2d' in files[v]:
        files2d.append(files[v])
    else:
        files3d.append(files[v])
    v = v + 1

print(files2d)
print(files3d)

date_files_2d = []
date_files_3d = []
i = 0
# taking dates from 2d fiels
while i < len(files2d):
    date_files_2d.append(files2d[i][-12:-4])
    i = i + 1

i = 0
# taking dates fro 3d files
while i < len(files3d):
    date_files_3d.append(files3d[i][-12:-4])
    i = i + 1

print(date_files_3d)
print(date_files_2d)

# finding comon dates in the 2d and 3d files
same_dates = list((Counter(date_files_2d) & Counter(date_files_3d)).elements())
print('sames')
print(same_dates)
# extracting dates from cleardays.csv
cleardays = pd.read_csv('cleardays_2013.csv')
Dates = cleardays['date'].tolist()

# converting list to list of stringS
map_sunnyDates = map(str, Dates)

str_sunny = list(map_sunnyDates)
print('sunny')
print(str_sunny)

# finding common dates in files and cleardays
Same_dates_2 = list((Counter(str_sunny) & Counter(same_dates)).elements())

print(Same_dates_2)

src = 'C:/Users/Jokin Lasa Escobales/Desktop/Data/Master/Thesis/MERRA2/MERRA2/'
dest = 'MY_MERRA_2013/'

ncfiles = os.listdir(src)
x = 0
filenumber = 0
print(len(Same_dates_2))

print(len(ncfiles))
while x < len(Same_dates_2):
    for file_name in ncfiles:
        print(file_name)
        print(Same_dates_2[x])
        if Same_dates_2[x] in file_name and filenumber < 1166:
            print('iguales')
            shutil.copy(src + file_name, dest + file_name)
            filenumber = filenumber + 1
    x = x + 1

