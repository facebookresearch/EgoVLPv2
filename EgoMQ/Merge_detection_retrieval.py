# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_path',
    type=str)

args = parser.parse_args()

result_det, result_rev = None, None

if os.path.exists(f'{args.output_path}/detections_postNMS.json'):
    result_det = f'{args.output_path}/detections_postNMS.json'
else:
    raise ValueError('Detection results missing.')

if os.path.exists(f'{args.output_path}/retreival_postNMS.json'):
    result_rev = f'{args.output_path}/retreival_postNMS.json'
else:
    raise ValueError('Retrieval results missing.')


with open(result_det, 'r') as fobj:
    data_det = json.load(fobj)

with open(result_rev, 'r') as fobj:
    data_rev = json.load(fobj)

data_submission = {"version": "1.0","challenge": "ego4d_moment_queries"}
data_submission['detect_results'] = data_det['results']
data_submission['retrieve_results'] = data_rev['results']

with open(f"{args.output_path}/submission.json", "w") as fp:
    json.dump(data_submission,fp)

print('Submission succesfully packed. Good luck!')
