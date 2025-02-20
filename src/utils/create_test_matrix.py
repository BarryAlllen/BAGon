import numpy as np

random_array = np.random.randint(0, 2, size=(10000, 30))

np.save('test_matrix.npy', random_array)

loaded_array = np.load('test_matrix.npy')
print("matrix shape: ", loaded_array.shape)