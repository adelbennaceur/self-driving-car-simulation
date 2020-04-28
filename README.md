
# self-driving-car-simulation
Self driving car simulation using hte provided simulator by Udacity.The goal of this project is to make a car drive autonomously using  a deep learning approach by feeding an input image to a neural network and predicting the steering angle associated with it.

code structure: 

#dataset:

#Dataset directory structure:
Directory
├── driving_log.csv
└─┬ IMG
  └── center_2019_03_12_17_11_43_382.jpg
  └── .....


#install dependecies:
```
$pip install -r requirement.txt
```

#train on your own dataset:
run the command to see the available arguments. Example: 
```
$ python main.py -dir your_data_directory -optimizer rmsprop -lr 0.001 -batch_size 32 -epochs 20 
```
