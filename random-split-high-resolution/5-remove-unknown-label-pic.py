import json
import os
import shutil

labels = ['nerve',
'fibrous annulus',
'nucleus pulposus',
'ligamentum flavum',
'extra-dural fat',
'muscle',
'lamina',
'tool1',
'tool2',
'tool3',
'tool4',
'tool5',
'tool6',
'tool7']

json_path = "json/"

json_list = os.listdir(json_path)

json_content = None

file_with_unknown_label = {}

for i in json_list:
	with open(os.path.join(json_path, i), encoding="utf-8") as f:
		json_content = json.load(f)
	for j in json_content["shapes"]:
		if j["label"] not in labels:
			file_with_unknown_label[i] = 1 #防止一个文件中有多个不认识的label

if not os.path.exists("unknown_label"):
	os.mkdir("unknown_label")

count = 0
for i in file_with_unknown_label:
	shutil.move(os.path.join(json_path, i),
		os.path.join("unknown_label", i))
	count += 1

print("{} files have unknown label. {} files moved.".format(
	len(file_with_unknown_label),
	count)
)