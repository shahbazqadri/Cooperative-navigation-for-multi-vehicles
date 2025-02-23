import airsim
import cv2
import numpy as np
import os
import pprint
import setup_path 
import tempfile

drone_names = ["Drone1", "Drone2", "Drone3", "Drone4", "Drone5"]

# connect to the AirSim simulator and setup the drones
client = airsim.MultirotorClient() 
client.confirmConnection()
for name in drone_names:
    client.enableApiControl(True, name)
    client.armDisarm(True, name)


# Wait for user input to takeoff
airsim.wait_key('Press any key to takeoff')
controllist = []
for i, name in enumerate(drone_names):
    f = client.takeoffAsync(vehicle_name=name)
    controllist.append(f)  
for i in range(len(drone_names)):
    controllist[i].join()


# Print the state of the drones
for name in drone_names:
    state = client.getMultirotorState(vehicle_name=name)
    s = pprint.pformat(state)
    print("state: %s" % s)



############################### Control Logic Goes Below ###############################

airsim.wait_key('Press any key to move vehicles')
for i, name in enumerate(drone_names):
    # pitch = 0.2 + 0.05 * i
    # roll = 0.1 + 0.02 * i
    # yaw_rate = 0.5 + 0.1 * i
    throttle = 0.6 + 0.05 * i
    controllist[i] = client.moveByRollPitchYawrateThrottleAsync(pitch=0.0, roll=0.0, yaw_rate=0.0, throttle=throttle, duration=5, vehicle_name=name)
for i in range(len(drone_names)):
    controllist[i].join()

airsim.wait_key('Press any key to reset to original state')
########################################################################################

# Reset the drones
for name in drone_names:
    client.armDisarm(False, name)
client.reset()
for name in drone_names:
    client.enableApiControl(False, name)


