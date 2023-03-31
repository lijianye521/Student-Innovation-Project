import json, os

json_path = "train"

file_list = os.listdir(json_path)

with_muscle = []

for i in file_list:
	content = None
	with open(json_path + "/" + i, encoding="utf-8") as f:
		content = json.load(f)
	for j in content["shapes"]:
		#if j['label'] == 'muscle' or j['label'] == 'muscles':
		if j['label'] == 'tool6':
			print(j['label'])
			with_muscle.append(i)

print(with_muscle)