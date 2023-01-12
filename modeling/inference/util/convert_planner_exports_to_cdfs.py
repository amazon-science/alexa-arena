# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import json
import os 

files = os.listdir('../planner_exports_combined')
write_dir = '../cdfs/'
for file in files:
    data = json.load(open('../planner_exports_combined/' + file))
    del data['command_batch']
    print("Writing " + write_dir + file)
    with open(write_dir + file, 'w') as outfile:
        json.dump(data, outfile)

   

    