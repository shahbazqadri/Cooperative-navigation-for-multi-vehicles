### Swarm Simulator

## Demo

https://github.com/user-attachments/assets/4a05d60a-57dd-41b3-82b6-68c9e6b604c7

## Setup

First, Navigate into the directory
```
$ cd ./swarm_simulator
```
Then, create the virtual environment (or if using Anaconda or conda, install as you 
would with the requirements.in file.) I dont know how to use Anaconda, so I will show
how its done in venv
```
$ python3 -m venv .venv
$ source ./.venv/bin/activate
$ pip3 install -r requirements.in
```
No error messages should occur. If Warnings occur, that should be okay. There is one version warning that is not critical or applicable to what we are doing.

## Running

To run the program, navigate into the sim_pkg directory
```
$ cd ./sim_pkg
```
Then, run the coachbot simulator program with the proper batch file
```
$ python3 coachbot_simulator.py -b custom_batch.json
```

## Changing the number of Landmarks or drones

To change the number of Landmarks or Drones, a few files must be edited. First, the starting location of the landmarks and drones must be edited in both the isam3_2.py file and the init_pose.py file. If the number of landmarks plus the number of drones is changed, then the NUMBER_OF_ROBOTS variable must be changed in the custom.json file. The changes that need to be made to init_pose.py and isam3_2.py are outlined in comments in those files. 
