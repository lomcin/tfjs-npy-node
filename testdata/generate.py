import numpy as np

np.save("1.npy", [1.5, 2.5])
np.save("2.npy", [[1.5, 43], [13, 2.5]])
np.save("3.npy", [[[1, 2, 3], [4, 5, 6]]])
np.save("4.npy", np.array([0.1, 0.2], "float32"))
np.save("uint8.npy", np.array([0, 127], "uint8"))

