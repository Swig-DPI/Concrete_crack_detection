### The purpose of this program is to split SDNET2018 data into test and training sets
'''
Created by: Scott Wigle
Purpose:
Split SDNET2018 data into test, train, and hold sets
It will create folders in the current directory that will have test train and hold data.


'''




import os
import pandas as pd
import numpy as np
import shutil
from sklearn.utils import shuffle


source_path = 'data/SDNET2018/' # This path should point to the downloaded SDNET2018 data set
dest_path = 'data/test_train_hold_1/' # this path will create a directory to store the split files
dest_folder = ['train', 'test', 'hold'] #  This is the folders which will be created
dest_folder_sub_dir = ['crack','NO_crack'] # each folder will have these two sub folders.

## Put local file sources into pandas datframe for splitting and movement
print('Sorting files for split files ')
data = []
for folder in sorted(os.listdir(source_path)):
    for sub_folder in sorted(os.listdir(source_path+folder)):
            for file in sorted(os.listdir(source_path+folder+'/'+sub_folder)):
                data.append((folder, sub_folder, file))

df = pd.DataFrame(data, columns=['Folder','sub_Folder', 'File'])


#  Add labels for data: 1 is crack, 0 is no crack label
df['label'] = df['sub_Folder'].apply(lambda x: 1 if 'C' in x else 0)

# put full path into dataframe
df['FullPath'] = df['Folder']+'/'+df['sub_Folder']+'/'+df['File']

# split into crack no_crack data frames and shuffle
dfc = df[df['label']== 1]
dfc = shuffle(dfc, random_state=42)
dfnc = df[df['label']== 0]
dfnc = shuffle(dfnc, random_state=42)

# split into even data frames for testing, training and holdout sets.
print('Splitting files')
dfc_split = np.array_split(dfc,3)
dfnc_split = np.array_split(dfnc,3)

# Combine into single pandas arrays
# dftrain = pd.concat([dfc_split[0],dfnc_split[0]])
# dftest = pd.concat([dfc_split[1],dfnc_split[1]])
# dfhold = pd.concat([dfc_split[2],dfnc_split[2]])

## Make directories
print('Creating Folders')
for f in dest_folder:
    for sd in dest_folder_sub_dir:
        directory = dest_path+f+'/'+sd
        print('Folder created in curret directory:  ',directory)
        if not os.path.exists(directory):
            os.makedirs(directory)



## Move crack files into dir
print('Copying crack files to created folders')
for idx1, df_i in enumerate(dfc_split):
    for idx2, df_row in df_i.iterrows():
        shutil.copy2(source_path+df_row['FullPath'], dest_path+dest_folder[idx1]+'/'+dest_folder_sub_dir[0])


## Move no crack files into dir

print('Copying no crack files to created folders')
for idx1, df_i in enumerate(dfnc_split):
    count = 0
    for idx2, df_row in df_i.iterrows():
        shutil.copy2(source_path+df_row['FullPath'], dest_path+dest_folder[idx1]+'/'+dest_folder_sub_dir[1])
        # this is to ensure an even split
        if count >= 2826:
            break
        count = count + 1

print('Completed')
