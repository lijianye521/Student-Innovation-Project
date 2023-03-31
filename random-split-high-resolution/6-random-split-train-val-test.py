import os, shutil, random

json_path = "json/"

file_list = os.listdir(json_path)

json_num = len(file_list)

split_map = [0 for _ in range(json_num)]

if not os.path.exists("train"):
	os.mkdir("train")
if not os.path.exists("val"):
	os.mkdir("val")
if not os.path.exists("test"):
	os.mkdir("test")

train_proportion = 0.4
val_proportion = 0.2
test_proportion = 0.4

train_num = int(json_num * train_proportion)
test_num = int(json_num * test_proportion)
val_num = int(json_num * val_proportion)

train_path = "train/"
val_path = "val/"
test_path = "test/"

NO_DATASET = 0
TRAIN_DATA = 1
VAL_DATA = 2
TEST_DATA = 3

temp = 0

while temp < train_num:
	randnum = random.randint(0, json_num - 1)
	if split_map[randnum] == NO_DATASET:
		split_map[randnum] = TRAIN_DATA        #means this file is regarded as train data
		temp += 1

temp = 0

while temp < val_num:
	randnum = random.randint(0, json_num - 1)
	if split_map[randnum] == NO_DATASET:
		split_map[randnum] = VAL_DATA        #means this file is regarded as train data
		temp += 1

temp = 0
while temp < test_num:
	randnum = random.randint(0, json_num - 1)
	if split_map[randnum] == NO_DATASET:
		split_map[randnum] = TEST_DATA        #means this file is regarded as train data
		temp += 1

count = 0

for i in range(json_num):
	if split_map[i] == TRAIN_DATA:
		shutil.move(os.path.join(json_path, file_list[i]),
			os.path.join(train_path, file_list[i]))
		count += 1

	elif split_map[i] == VAL_DATA:
		shutil.move(os.path.join(json_path, file_list[i]),
			os.path.join(val_path, file_list[i]))
		count += 1

	elif split_map[i] == TEST_DATA:
		shutil.move(os.path.join(json_path, file_list[i]),
			os.path.join(test_path, file_list[i]))
		count += 1

	else:
		print("No dataset was assigned, json file", count + 1)


print("{} files moved. \n{} train files.\n{} test files.\n{}val files.".format(
	count, train_num, test_num, val_num)
)