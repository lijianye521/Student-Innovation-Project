import os

rename_list = {
	"周斌": "zhoubin",
	"赵晓慧": "zhaoxiaohui",
	"朱恒": "zhuheng",
	"鲍雪冬": "baoxuedong",
	"王传合": "wangchuanhe", 
	"沈文堂": "shenwentang", 
	"李明": "liming", 
	"鞠洪法": "juhongfa", 
	"江英美": "jiangyingmei",
	"杜长亮": "duchangliang", 
	"董祥文": "dongxiangwen", 
	"陈欣": "chenxin", 
	"陈胜利": "chenshengli"
}

json_path = "json/"
pic_path = "pic/"

file_list = os.listdir(json_path)

file_num = 0

for i in file_list:
	for j in rename_list:
		if j in i:
			new_name = i.replace(j, rename_list[j])
			os.rename(
				os.path.join(json_path, i),
				os.path.join(json_path, new_name)
				)
			file_num += 1

file_list2 = os.listdir(pic_path)
for i in file_list2:
	for j in rename_list:
		if j in i:
			new_name = i.replace(j, rename_list[j])
			os.rename(
				os.path.join(pic_path, i),
				os.path.join(pic_path, new_name)
				)
			file_num += 1

print("共处理了{}个文件。".format(file_num))