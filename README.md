
# self-driving-car-simulation
Self driving car simulation using Udacity.The goal of this project is to make a car drive autonomously using a deep learning approach by feeding an input image to a neural network and predicting the steering angle.

## Code structure: 
* [`main.py`](https://github.com/adelbennaceur/self-driving-car-simulation/blob/master/main.py) : contains the script to train the neural network.
* [`model.py`](https://github.com/adelbennaceur/self-driving-car-simulation/blob/master/model.py): contains the model (neural network) architecture.
*  [`utils.py`](https://github.com/adelbennaceur/self-driving-car-simulation/blob/master/utils.py): contains  functions for preprocessing and loading the data.
*  [`drive.py`](https://github.com/adelbennaceur/self-driving-car-simulation/blob/master/drive.py): contains the script to connect to the simulator and run your model.

## Model architecture 
the model is inspired from the nvidia End to End self driving car network.
![CNN architecture](imgs/cnn-architecture.png?raw=True)

## Dataset:
The dataset is collected from the udacity self driving car simulator.

## Dataset directory structure:
```
Directory
├── driving_log.csv
└─┬ IMG
  └── center_2019_03_12_17_11_43_382.jpg
  └── .....
```
## Results
![Results](imgs/example.gif?raw=True)
## Install dependencies:
```
$pip install -r requirements.txt
```



## Train on your own dataset:
run the command to see the available arguments:
```
$ python main.py -h
```
### Example: 
```
$ python main.py -dir path/to/dataset -optimizer rmsprop -lr 0.001 -batch_size 32 -epochs 20 
```
