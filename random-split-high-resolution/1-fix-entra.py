import os, json

dirs = ["赵晓慧", "周斌", "朱恒", "王传合", "沈文堂", "李明", "鞠洪法", "江英美",
	"杜长亮", "董祥文", "陈欣", "陈胜利", "鲍雪冬"]

def check_if_have_entra(file_path):
	file_need_write = False
	content = json.load(open(file_path, encoding="utf-8"))
	for i in content["shapes"]:
		if i["label"] == "entra-dural fat":
			i["label"] = "extra-dural fat"
			file_need_write = True

	if file_need_write:
		f = open(file_path, "w", encoding="utf-8")
		json.dump(content, f)
		print(file_path)

for i in dirs:
	dir_list = os.listdir(i)
	for j in dir_list:
		dir_name = os.path.join(i, j)
		sub_dir_list = os.listdir(dir_name)
		for k in sub_dir_list:
			if k.split(".")[-1] == "json":
				file_path = os.path.join(dir_name, k)
				check_if_have_entra(file_path)

