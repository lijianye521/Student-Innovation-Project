import os, json, shutil

dirs = ["赵晓慧", "周斌", "朱恒", "王传合", "沈文堂", "李明", "鞠洪法", "江英美",
	"杜长亮", "董祥文", "陈欣", "陈胜利", "鲍雪冬"]

pic_dir = "pic/"
json_dir = "json/"

if not os.path.exists(pic_dir):
	os.mkdir(pic_dir)

if not os.path.exists(json_dir):
	os.mkdir(json_dir)

def move_file(file_path):
	if file_path.split(".")[-1] == "jpg":
		shutil.move(file_path, pic_dir)
	elif file_path.split(".")[-1] == "json":
		shutil.move(file_path, json_dir)

for i in dirs:
	dir_list = os.listdir(i)
	for j in dir_list:
		dir_name = os.path.join(i, j)
		sub_dir_list = os.listdir(dir_name)
		for k in sub_dir_list:
			file_path = os.path.join(dir_name, k)
			move_file(file_path)

