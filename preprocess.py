from dpu_utils.utils import RichPath
import json
import numpy as np
import sys
import random

assert sys.argv[1].endswith(".jsonl.gz")

jsonl = RichPath.create(sys.argv[1]).read_by_file_suffix()
size = None
total = 0

for datapoint in jsonl:
	v = np.array(datapoint["Property"])
	if size is None:
		size = v.shape[0]
		counter = np.zeros((size, ))
	counter += v
	total += 1

threshold = 0.2
ratios = counter / total
print("ratios = {}".format(ratios))

keep_idx = []
for i in range(0, len(ratios)):
	if not (ratios[i] - 0 <= threshold or 1 - ratios[i] <= threshold):
		keep_idx.append(i)

print("Previous Size = {}, After Size = {}".format(len(ratios), len(keep_idx)))

jsonl = RichPath.create(sys.argv[1]).read_by_file_suffix()

with open("data/train.jsonl", 'w') as fd_train, \
	open("data/valid.jsonl", 'w') as fd_valid, \
	open("data/test.jsonl", 'w') as fd_test,:
	for datapoint in jsonl:
		v = np.array(datapoint["Property"])
		new_prop = []
		for i in keep_idx:
			new_prop.append(datapoint["Property"][i])
		datapoint["Property"] = new_prop
		r = random.random()
		if r <= 0.8:
			print(json.dumps(datapoint), file=fd_train)
		elif r > 0.9:
			print(json.dumps(datapoint), file=fd_test)
		else:
			print(json.dumps(datapoint), file=fd_valid)
