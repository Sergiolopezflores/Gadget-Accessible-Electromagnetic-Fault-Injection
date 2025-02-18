To create the dataset we have to follow a series of steps:

# 1 Transform the capture to csv.
The first thing to do is to transform the capture we have made to a csv format to be able to work with it.
To do this we use the file **create_csv.py**. This file takes the data from the capture and groups them in blocks of 1000 together with label that identifies the type of malware to which the captures belong.
This data is saved in a csv file. This should be done with all the captures we have.

# 2 Create a final CSV
Once we have created all the csv's of the captures that we have, what we have to do is to group all those csv's in only one to have all the captures of a category (type, family...) in only one csv file, that later will be the one that we will
transform to one-hot format.
For this we use the file **create_csv_final.py**.

# 3 Create one-hot-encoding file
The data that we are going to use to train the models will be in one-hot-encoding format. To create this file we use the script **create_one_hot_encoding.py**, that what it does is, from the csv_final that we have, it transforms it
to one-hot-encoding format.

# 4 Validate one-hot-encoding file
Once we have the one-hot-encoding file, we can use the script **validate_csv.py** to check that the file has been created correctly.
