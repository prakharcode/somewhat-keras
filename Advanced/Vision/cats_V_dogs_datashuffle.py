##########################
#    Data Preprocessing  #
##########################
import os
import shutil

cwd = os.getcwd()
data_dir = os.path.join(cwd, data)

os.mkdir(os.path.join(data_dir, cats_V_dogs))
cats_V_dogs_dir = os.path.join(data_dir, cats_V_dogs)

train = os.path.join(cats_V_dogs, train)
test = os.path.join(cats_V_dogs, train)
val = os.path.join(cats_V_dogs, train)
os.mkdir(train)
os.mkdir(test)
os.mkdir(val)

actual_train_dog_dir = os.path.join(train, dogs)
actual_train_cat_dir = os.path.join(train, cats)
actual_val_dog_dir = os.path.join(val, dogs)
actual_val_cat_dir = os.path.join(val, cats)
actual_test_dog_dir = os.path.join(test, dogs)
actual_test_cat_dir = os.path.join(test, cats)
os.mkdir(actual_train_dog_dir)
os.mkdir(actual_train_cat_dir)
os.mkdir(actual_val_dog_dir)
os.mkdir(actual_val_cat_dir)
os.mkdir(actual_test_dog_dir)
os.mkdir(actual_test_cat_dir)

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test1')

cat_fnames_train = ['cat.{}.jpg'.format(i) for i in range(1000)]
cat_fnames_val = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
cat_fnames_test = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]

for fname in cat_fnames_train:
    src = os.path.join(train_dir, fname)
    des = os.path.join(actual_train_cat_dir, fname)
    shutil.move(src, des)

for fname in cat_fnames_val:
    src = os.path.join(train_dir, fname)
    des = os.path.join(actual_val_cat_dir, fname)
    shutil.move(src, des)

for fname in cat_fnames_test:
    src = os.path.join(train_dir, fname)
    des = os.path.join(actual_test_cat_dir, fname)
    shutil.move(src, des)

dog_fnames_train = ['dog.{}.jpg'.format(i) for i in range(1000)]
dog_fnames_val = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
dog_fnames_test = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]

for fname in dog_fnames_train:
    src = os.path.join(train_dir, fname)
    des = os.path.join(actual_train_dog_dir, fname)
    shutil.move(src, des)

for fname in dog_fnames_val:
    src = os.path.join(train_dir, fname)
    des = os.path.join(actual_val_dog_dir, fname)
    shutil.move(src, des)

for fname in dog_fnames_test:
    src = os.path.join(train_dir, fname)
    des = os.path.join(actual_test_dog_dir, fname)
    shutil.move(src, des)

print(" S A N I T Y C H E C K")
print(f"Total dogs train image: {len(os.listdir(actual_train_dog_dir))}")
print(f"Total dogs val image: {len(os.listdir(actual_val_dog_dir))}")
print(f"Total dogs test image: {len(os.listdir(actual_test_dog_dir))}")
print(f"Total cats train image: {len(os.listdir(actual_train_cat_dir))}")
print(f"Total cats val image: {len(os.listdir(actual_val_cat_dir))}")
print(f"Total cats test image: {len(os.listdir(actual_test_cat_dir))}")
