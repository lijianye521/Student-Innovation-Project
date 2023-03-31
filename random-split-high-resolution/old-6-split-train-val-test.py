import os, shutil

json_path = "json/"

file_list = os.listdir(json_path)

if not os.path.exists("train"):
	os.mkdir("train")
if not os.path.exists("val"):
	os.mkdir("val")
if not os.path.exists("test"):
	os.mkdir("test")

#rate: 2trian: 1val: 2test
#sample method: for each 5 files: train, test, val train, test

count = 0
train_num = 0
test_num = 0
val_num = 0

train_path = "train/"
val_path = "val/"
test_path = "test/"

for i in file_list:

	if count % 5 == 0 or count % 5 == 3:
		shutil.move(os.path.join(json_path, i),
			os.path.join(train_path, i))
		train_num += 1
	elif count % 5 == 1 or count % 5 == 4:
		shutil.move(os.path.join(json_path, i),
			os.path.join(test_path, i))
		test_num += 1
	else:
		shutil.move(os.path.join(json_path, i),
			os.path.join(val_path, i))
		val_num += 1

	count += 1

print("{} files moved. \n{} train files.\n{} test files.\n{}val files.".format(
	count, train_num, test_num, val_num)
)