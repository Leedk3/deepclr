import math
import numpy as np

def reverseXcYcZc(calib_info, method):
    tx = calib_info["Tx"]
    ty = calib_info["Ty"]
    tz = calib_info["Tz"]

    if method == "lidar": 
        rx = calib_info["Yaw"] * math.pi / 180.0
        ry = calib_info["Roll"] * math.pi / 180.0
        rz = calib_info["Pitch"] * math.pi / 180.0
    elif method == "camera" :
        rx = calib_info["Rx"] * math.pi / 180.0
        ry = calib_info["Ry"] * math.pi / 180.0
        rz = calib_info["Rz"] * math.pi / 180.0
    else :
        print("method is wrong")
        return np.array([])

    s1 = math.sin(rx)
    s2 = math.sin(ry)
    s3 = math.sin(rz)

    c1 = math.cos(rx)
    c2 = math.cos(ry)
    c3 = math.cos(rz)
    
    # print(s1, s2, s3, c1, c2, c3)

    # (3 x 1)
    Twc = np.array([[tx], [ty], [tz]])
    
    # (3 x 3)
    Rwc = np.array([
         [c2 * s3, -s2, c2 * c3],
         [-c1 * c3 + s1 * s2 * s3, s1 * c2, c1 * s3 + s1 * s2 * c3],
         [-s1 * c3 - c1 * s2 * s3, -c1 * c2, s1 * s3 - c1 * s2 * c3]])
    # print(Twc.shape, Rwc.shape)

    minus_RwcT = -1 * np.transpose(Rwc)

    # minus_RwcT = np.array(
    #     [-Rwc[0][0], -Rwc[1][0], -Rwc[2][0]],
    #     [-Rwc[0][1], -Rwc[1][1], -Rwc[2][1]],
    #     [-Rwc[0][2], -Rwc[1][2], -Rwc[2][2]])

    right_value = np.matmul(minus_RwcT, Twc)

    total_value = np.array([
        [Rwc[0][0], Rwc[1][0], Rwc[2][0], right_value[0][0]],
        [Rwc[0][1], Rwc[1][1], Rwc[2][1], right_value[1][0]],
        [Rwc[0][2], Rwc[1][2], Rwc[2][2], right_value[2][0]],
        [0, 0, 0, 1],
    ])

    return total_value

    # const result_4dim = [result[0], result[1], result[2], [1]];
    # const xyz1 = math.multiply(math.inv(total_value), result_4dim);
    # // const result = this.multiplyMatrices(total_value, [[X], [Y], [Z], [1]]);
    # // 카메라 회전 값

    # return xyz1;
