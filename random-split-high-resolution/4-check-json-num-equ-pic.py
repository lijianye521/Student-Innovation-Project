import os, shutil

pic_path = "pic/"
json_path = "json/"

pic_files = {}
json_files = {}

pic_list = os.listdir(pic_path)
json_list = os.listdir(json_path)

files_not_matched = []

moved_file_num = 0

for i in pic_list:
	if i.split(".")[0] + ".json" not in json_list:
		files_not_matched.append(i)

if not os.path.exists("not_matched"):
	os.mkdir("not_matched")
for i in files_not_matched:
	shutil.move(os.path.join(pic_path, i),
		os.path.join("not_matched", i))
	moved_file_num += 1

for i in files_not_matched:
	print(i)
print("now moved file num:", moved_file_num)

files_not_matched = []

for i in json_list:
	if i.split(".")[0] + ".jpg" not in pic_list:
		files_not_matched.append(i)
		
for i in files_not_matched:
	shutil.move(os.path.join(json_path, i),
			os.path.join("not_matched", i))
	moved_file_num += 1

for i in files_not_matched:
	print(i)

print("total moved file num:", moved_file_num)