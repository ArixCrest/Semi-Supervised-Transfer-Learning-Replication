image_paths = []
with open('images.txt') as f:
    image_paths = f.readlines()
    for i in range(len(image_paths)):
        image_paths[i] = image_paths[i].split()[-1]
f.close()
import shutil
import os
with open('train_test_split.txt') as f:
    train_test = f.readlines()
    for i in range(len(train_test)):
        typ = int(train_test[i].split()[-1])
        if(typ==0):
            directory = './train/'+image_paths[i].split('/')[0]+'/'
        else:
            directory = './test/'+image_paths[i].split('/')[0]+'/'
        os.makedirs(directory, exist_ok = True)
        shutil.move('./images/'+image_paths[i],
                    directory)
