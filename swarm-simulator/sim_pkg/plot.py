import csv
import numpy as np
import matplotlib.pyplot as plt

nb_agents = 5

def csv_to_array_of_arrays(file_path):
    arrays = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert each row to a numerical array
            numerical_array = np.array([float(i) for i in row])
            arrays.append(numerical_array)
    # Convert the list of arrays to a 2D NumPy array
    return np.vstack(arrays).T  # Transpose to match the shape expected by the plotting code


for i in range(5):
    arr1 = csv_to_array_of_arrays(f"./obsv_x{i}_est.csv")
    arr2 = csv_to_array_of_arrays(f"./obsv_x{i}_true.csv")
    plt.plot(arr1[0], arr1[1], label='estimated Vehicle ' + str(i))
    plt.plot(arr2[0], arr2[1], label='true Vehicle ' + str(i))
plt.legend()
plt.title('Vehicle trajectories')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

