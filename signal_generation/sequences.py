
import numpy as np
import math
"""
https://web.archive.org/web/20110806114215/http://homepage.mac.com/afj/taplist.html
https://in.ncu.edu.tw/ncume_ee/digilogi/prbs.htm

4-bits:
- 2 taps: [4,3]

5-bits:
- 2 taps: [5,3]
- 4 taps: [5,4,3,2], [5,4,3,1]

6-bits:
- 2 taps: [6,5]
- 4 taps: [6,5,4,1], [6,5,3,2]

7-bits:
- 2 taps: [7,6], [7,4]
- 4 taps: [7,6,5,4], [7,6,5,2], [7,6,4,2], [7,6,4,1], [7,5,4,3]
- 6 taps: [7,6,5,4,3,2], [7,6,5,4,2,1]

8-bits:
- 4 taps: [8,7,6,1], [8,7,5,3], [8,7,3,2], [8,6,5,4], [8,6,5,3], [8,6,5,2]
- 6 taps: [8,7,6,5,4,2], [8,7,6,5,2,1]

9-bits:
- 2 taps: [9,5] 
- 4 taps: [9,8,7,2], [9,8,6,5], [9,8,5,4], [9,8,5,1], [9,8,4,2], [9,7,6,4], [9,7,5,2], [9,6,5,3] 
- 6 taps: [9,8,7,6,5,3], [9,8,7,6,5,1], [9,8,7,6,4,3], [9,8,7,6,4,2], [9,8,7,6,3,2], [9,8,7,6,3,1], [9,8,7,6,2,1], 
          [9,8,7,5,4,3], [9,8,7,5,4,2], [9,8,6,5,4,1], [9,8,6,5,3,2], [9,8,6,5,3,1], [9,7,6,5,4,3], [9,7,6,5,4,2] 
- 8 taps: [9,8,7,6,5,4,3,1]
"""
def maximal_length_sequence(reg_length, taps):

    register = np.zeros([reg_length])
    register[0] = 1

    sr_size = 2**reg_length - 1
    SR = np.zeros([sr_size])

    for idx in range(sr_size):
        SR[idx] = register[-1]

        tmp_sum = 0
        for jdx in range(taps.shape[0]):
            tmp_sum += register[taps[jdx]]

        register[1:] = register[0:-1]
        register[0] = tmp_sum % 2

    SR = 2*SR - 1
    return SR
    
#------------------------------------------------------------------------------
def barker_code(code_length):

    if(code_length == 2):
        data = np.array([1, -1])
    elif(code_length == 3):
        data = np.array([1, 1, -1])
    elif(code_length == 4):
        data = np.array([1, 1, -1, 1])
    elif(code_length == 5):
        data = np.array([1, 1, 1, -1, 1])
    elif(code_length == 7):
        data = np.array([1, 1, 1, -1, -1, 1, -1])
    elif(code_length == 11):
        data = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1])
    elif(code_length == 13):
        data = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    else:
        data = np.array([1, 1, 1, -1, 1])
        print("code length supplied is not a valid barker code length.  Setting code length to 5")
        
    return data
