from numpy import array
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt

weights = array([[[[0.0968, -0.8211,  0.0251, -0.2740, -0.5661],
                   [0.4542, -0.4043, -0.1657, -0.3475, -0.2537],
                   [0.4953, -0.6625, -0.1406, -0.2940, -0.2165],
                   [0.4248, -0.3507,  0.0307, -0.2516,  0.1607],
                   [-0.0920, -0.8978, -0.4380, -0.4385,  0.6969]]],


                 [[[0.0624, -0.0793, -0.2120, -0.1464, -0.2170],
                   [-0.1021, -0.2673, -0.0287, -0.3517, -0.0767],
                     [-0.1061, -0.1728, -0.2756, -0.1496,  0.0306],
                     [-0.0571, -0.1779, -0.1944, -0.0145,  0.3807],
                     [-0.2370, -0.1913, -0.4341, -0.1277,  0.2312]]],


                 [[[0.2664,  0.1050,  0.2046, -0.4075, -0.0152],
                   [0.0865, -0.4831, -0.5307,  0.0152, -0.0534],
                     [-0.5785, -0.1440,  0.0282,  0.0390,  0.0536],
                     [-0.1875, -0.0035,  0.0436,  0.0713, -0.0535],
                     [-0.3628, -0.1510, -0.1777, -0.3055, -0.3866]]],


                 [[[-0.5098, -0.2288, -0.1403,  0.1277, -0.1582],
                   [-0.0613, -0.1119, -0.2797, -0.0804,  0.4871],
                     [-0.3829,  0.0957, -0.3622, -0.4436,  0.0933],
                     [-0.1080, -0.2812,  0.0400, -0.1630, -0.2876],
                     [-0.2009,  0.0495,  0.1397, -0.0583, -0.2105]]],


                 [[[-0.3273, -0.2617, -0.1021,  0.4604,  0.4314],
                   [-0.2829, -0.0980, -0.5014, -0.0545,  0.2941],
                     [-0.0290, -0.1571, -0.3528, -0.1762,  0.3559],
                     [-0.1699, -0.0905, -0.2249, -0.2388,  0.3417],
                     [-0.0222,  0.0896, -0.2860, -0.4146, -0.4674]]],


                 [[[-0.5577,  0.6081,  0.3114,  0.5095,  0.7077],
                   [-0.5752,  0.3106,  0.3188, -0.3955, -0.4994],
                     [-0.8702, -0.6875, -0.4816, -0.3299, -0.7751],
                     [-0.4054, -0.5528, -0.4070, -0.4337, -0.3860],
                     [0.6299,  0.3085, -0.4789, -0.0694,  0.1630]]],


                 [[[-0.2122, -0.1825,  0.2091, -0.0282, -0.0565],
                   [-0.1402, -0.0544,  0.0135, -0.2169, -0.1801],
                     [-0.1080,  0.0091,  0.0251, -0.1476,  0.1087],
                     [-0.1346,  0.0787, -0.1185, -0.1745, -0.0647],
                     [-0.3487, -0.0919,  0.0131, -0.0607, -0.0863]]],


                 [[[0.4546, -0.2556, -0.1561, -0.1312, -0.4787],
                   [0.0814, -0.0879, -0.1060, -0.2951,  0.0442],
                     [-0.4288, -0.2032, -0.0564,  0.0810,  0.0744],
                     [0.1142, -0.1167, -0.3032, -0.1566, -0.1937],
                     [-0.0625, -0.0458,  0.0411, -0.0293, -0.3262]]],


                 [[[-0.1121, -0.4014, -0.3718, -0.1007,  0.3528],
                   [-0.1558, -0.1988, -0.2056, -0.0039, -0.0714],
                     [-0.0183,  0.0248, -0.2611, -0.1393, -0.1233],
                     [-0.0409,  0.0311, -0.2669, -0.1730, -0.1891],
                     [-0.1449,  0.1711, -0.4317, -0.3438,  0.2263]]],


                 [[[-0.3885, -0.3159,  0.0468, -0.2213, -0.0319],
                   [-0.4887, -0.1659,  0.1357,  0.4591, -0.1193],
                     [-0.3398, -0.1824, -0.0191,  0.3852, -0.1134],
                     [-0.3534, -0.4404, -0.2168,  0.1593, -0.3471],
                     [0.0392, -0.3225, -0.3158, -0.6675, -0.5050]]]])

bias = array([-1.5571, -1.3520, -1.7155, -1.3240, -1.3753, -1.9704, -0.8887, -1.6161,
              -1.4209, -0.8150])

test_array = np.empty([250, 2], dtype=float)

for i in range(weights.shape[0]):
    w = weights[i][0].reshape(25)
    for j in range(25):
        test_array[j+i*25][0] = w[j]
        test_array[j+i*25][1] = bias[i]

print(test_array)


# b = np.ones(10)
# features = np.column_stack((features, b))

# whitened = whiten(features)

# codebook, distortion = kmeans(whitened, 4)
# print(codebook, distortion)

plt.scatter(test_array[:, 0], test_array[:, 1])
# plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
plt.show()