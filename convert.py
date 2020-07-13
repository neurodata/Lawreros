from numpy import genfromtxt
from math import factorial
import numpy as np

for sub in range(26044,26046):
    for ses in range(1,46):
        file_name = f'sub-00{sub}_ses-{ses}'

        my_data = genfromtxt(f'/Users/ross/Documents/discrim_test/Corrected/IPCAS_6/ORIG/{file_name}.csv', delimiter=',', skip_header=1)

        print('oof')

        a = sum(range(1, len(my_data)))
        arr = np.zeros((a,3))
        z=0
        for num in range(len(my_data)):
            for i in range(len(my_data[num])):
                if i > num:
                    #print(f'{num+1} {i+1} {my_data[num][i]}')
                    arr[z][0]= f'{num+1}'
                    arr[z][1]= f'{i+1}'
                    arr[z][2] = my_data[num][i]
                    z=z+1
        
        np.savetxt(f"/Users/ross/Documents/discrim_test/Corrected/IPCAS_6/NEW/{file_name}_dwi_desikan_space-MNI152NLin6_res-2x2x2_connectome.csv", arr,fmt='%d %d %f', delimiter=' ')
