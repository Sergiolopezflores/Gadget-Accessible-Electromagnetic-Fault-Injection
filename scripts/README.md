The first thing we have to do is to capture the electromagnetic waves emitted by the processor while different types of malware or goodware are running.

# 1 Execute malware and goodware
The first thing to do is to run different types of malware and goodware on the target device. To do this we use the script **generate_traces.py**, which does is to run different malware or goodware on the device.
To indicate what to run on the target device, it is passed a csv (there are some examples in this folder, depending on what is required), at the time of execution, which indicates where the different types of files to run are stored.

An example of the execution of this script could be:

```bash
python3 generate_traces.py ./cmdFile_packer.csv -c 1640000
```
Where the first parameter that we pass it is the csv where the different types of malware that we are going to execute in the device are found and with the option -c we pass it how many times we want it to be executed. We have to take into account that it has to last the sufficient time to make the complete capture that we want to make.


# 2 Capture traces
Once the malware or goodware is running on the target device, the electromagnetic signals emitted by the processor must be captured.
To do this we run the script **EMFDetector.ino** on the arduino. This file is configured to capture the traces at a frequency of 1KHz, which would be about 1000 samples per second. For this it is indicated that the time between readings is every 1ms. These traces are stored in a file, called **bashlite.txt** but can be renamed.



To create the dataset we have to follow a series of steps:

# 3 Transform the capture to csv.
The first thing to do is to transform the capture we have made to a csv format to be able to work with it.
To do this we use the file **create_csv.py**. This file takes the data from the capture and groups them in blocks of 1000 together with label that identifies the type of malware to which the captures belong.
This data is saved in a csv file. This should be done with all the captures we have.

# 4 Create a final CSV
Once we have created all the csv's of the captures that we have, what we have to do is to group all those csv's in only one to have all the captures of a category (type, family...) in only one csv file, that later will be the one that we will
transform to one-hot format.
For this we use the file **create_csv_final.py**.

# 5 Create one-hot-encoding file
The data that we are going to use to train the models will be in one-hot-encoding format. To create this file we use the script **create_one_hot_encoding.py**, that what it does is, from the csv_final that we have, it transforms it
to one-hot-encoding format.

# 6 Validate one-hot-encoding file
Once we have the one-hot-encoding file, we can use the script **validate_csv.py** to check that the file has been created correctly.




The **one-hot-encoding.txt** file follows the structure detailed below:


```text
30,7,11,21,15,49,49,21,16,20,12,45,103,23,...,1.0,0.0,0.0,0.0,0.0
31,5,46,31,37,30,27,37,39,9,120,45,78,11,...0.0,1.0,0.0,0.0,0.0
...
```



Each data instance consists of 1000 integers representing feature values, followed by *X* one-hot encoded values indicating the class label (e.g., goodware, ransomware), ensuring a structured and interpretable dataset.

To preserve the integrity of the dataset, each line must contain exactly 1000 feature values followed by *X* one-hot encoded class indicators. A validation script, available on the GitHub repository, verifies the adherence to this format, preventing errors due to malformed input data.
