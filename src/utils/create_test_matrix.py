import numpy as np

random_array = np.random.randint(0, 2, size=(10000, 30))

# np.save('test_matrix.npy', random_array)

loaded_array = np.load('./src/utils/test_matrix.npy')
print("matrix shape: ", loaded_array.shape)
print("matrix: \n", loaded_array)
print(f"fist line: {loaded_array[0]}")
print(f"fist line: {loaded_array[1]}")
print(f"fist line: {loaded_array[2]}")
print(f"fist line shape: {loaded_array[2,:].shape}")