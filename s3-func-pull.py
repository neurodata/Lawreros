import re
import os
from collections import OrderedDict
import m2g.scripts.m2g_cloud as cm2g
import m2g.utils.cloud_utils
from m2g.utils import gen_utils
from m2g.utils.cloud_utils import get_credentials
from m2g.utils.cloud_utils import get_matching_s3_objects
from numpy import genfromtxt
from math import factorial
import numpy as np

bucket = 'ndmg-data'
paths = ['IPCAS4/Py3_test']
localpaths = ['/IPCAS4'] #SWU4, BNU3

num=paths.__len__()

subj_pattern = r"(?<=sub-)(\w*)(?=/ses)"
sesh_pattern = r"(?<=ses-)(\d*)"

for qq in range(num):
    path=paths[qq]
    localpath=localpaths[qq]

    all_subfiles = get_matching_s3_objects(bucket, path + "/sub-", suffix="")

    subjs = list(set(re.findall(subj_pattern, obj)[0] for obj in all_subfiles))

    seshs = OrderedDict()
    client = m2g.utils.cloud_utils.s3_client(service="s3")

    for subj in subjs:
        prefix = f"{path}/sub-{subj}/"
        all_seshfiles = get_matching_s3_objects(bucket, prefix, "measure-correlation.csv")
        seshs = list(obj for obj in all_seshfiles)
        # sesh = list(set([re.findall(sesh_pattern, obj)[0] for obj in all_seshfiles]))

        # Load in all the files selected above for the given subject
        for csv in seshs:
            atlas = csv.split("/")[-2]
            subsesh = f"{csv.split('/')[-9]}_{csv.split('/')[-8]}"

            os.makedirs(f"{localpath}/{atlas}/RAW", exist_ok=True)
            os.makedirs(f"{localpath}/{atlas}/NEW", exist_ok=True)

            client.download_file(bucket, f"{csv}", f"{localpath}/{atlas}/RAW/{subsesh}_measure-correlation.csv")
            print(f"Downloaded {csv}")

            my_data = genfromtxt(f"{localpath}/{atlas}/RAW/{subsesh}_measure-correlation.csv", delimiter=',', skip_header=1)

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
                
            np.savetxt(f"{localpath}/{atlas}/NEW/{subsesh}_measure-correlation.csv", arr,fmt='%d %d %f', delimiter=' ')
            print(f"Converted")



print('oof')
