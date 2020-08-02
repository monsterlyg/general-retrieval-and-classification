import os
LV_dataPath = "/mnt/sjk/data/traindata/fgvc/LV-single/Images"

dirNames = os.listdir(LV_dataPath)
class_id = 186
with open("/mnt/liyinggang/tmp/LV/Datasets/INSTRES1_train.txt", 'a') as file:
    with open("/mnt/liyinggang/tmp/LV/Datasets/INSTRES1_val.txt", 'a') as test:
        for dirName in dirNames:

            classPath = os.path.join(LV_dataPath, dirName)
            imgNames = os.listdir(classPath)
            internal = 0
            for imgName in imgNames:

                imgPath = os.path.join(classPath, imgName)
                if internal % 5 == 0:
                    test.write("{}*{}*{}\n".format(imgPath, class_id, dirName))
                else:
                    file.write("{}*{}*{}\n".format(imgPath, class_id, dirName))
                internal += 1
            class_id += 1
print('over')
