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
