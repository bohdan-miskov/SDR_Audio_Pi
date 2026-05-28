import numpy as np

def calculate_position(delays, mic_x, mic_y, mic_z, v=343.0):
    time_delays = [0] + delays 
    n = len(time_delays)
    
    eps = 1e-9
    for i in range(n):
        if time_delays[i] == 0 and i != 0:
            time_delays[i] = eps

    Amat = np.zeros(n)
    Bmat = np.zeros(n)
    Cmat = np.zeros(n)
    Dmat = np.zeros(n)

    for i in range(2, n):
        Amat[i] = (1 / (v * time_delays[i])) * (-2 * mic_x[0] + 2 * mic_x[i]) - \
                  (1 / (v * time_delays[1])) * (-2 * mic_x[1] + 2 * mic_x[2])
                  
        Bmat[i] = (1 / (v * time_delays[i])) * (-2 * mic_y[0] + 2 * mic_y[i]) - \
                  (1 / (v * time_delays[1])) * (-2 * mic_y[1] + 2 * mic_y[2])
                  
        Cmat[i] = (1 / (v * time_delays[i])) * (-2 * mic_z[0] + 2 * mic_z[i]) - \
                  (1 / (v * time_delays[1])) * (-2 * mic_z[1] + 2 * mic_z[2])

        Sum1 = (mic_x[0]**2 + mic_y[0]**2 + mic_z[0]**2) - (mic_x[i]**2 + mic_y[i]**2 + mic_z[i]**2)
        Sum2 = (mic_x[0]**2 + mic_y[0]**2 + mic_z[0]**2) - (mic_x[1]**2 + mic_y[1]**2 + mic_z[1]**2)
        
        Dmat[i] = v * (time_delays[i] - time_delays[1]) + \
                  (1 / (v * time_delays[i])) * Sum1 - \
                  (1 / (v * time_delays[1])) * Sum2

    M = np.zeros((n - 2, 3))
    D = np.zeros((n - 2, 1))

    for i in range(2, n):
        M[i-2, 0] = Amat[i]
        M[i-2, 1] = Bmat[i]
        M[i-2, 2] = Cmat[i]
        D[i-2] = -Dmat[i]

    Minv = np.linalg.pinv(M)
    T = np.dot(Minv, D)

    x_res, y_res, z_res = T[0][0], T[1][0], T[2][0]
    
    return x_res, y_res, z_res