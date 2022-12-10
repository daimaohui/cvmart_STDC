# _*_ coding:utf-8 _*_
from glob import glob
from random import randint
if __name__ == '__main__':
    files=[]
    file = glob('/home/data/1824/*.jpg')
    for i in file:
        files.append(i)
    file = glob('/home/data/1825/*.jpg')
    for i in file:
        files.append(i)
    file = glob('/home/data/1826/*.jpg')
    for i in file:
        files.append(i)
    file = glob('/home/data/1827/*.jpg')
    for i in file:
        files.append(i)
    file = glob('/home/data/572/*.jpg')
    for i in file:
        files.append(i)
    file = glob('/home/data/573/*.jpg')
    for i in file:
        files.append(i)
    file = glob('/home/data/652/*.jpg')
    for i in file:
        files.append(i)
    file = glob('/home/data/789/*.jpg')
    for i in file:
        files.append(i)
    train = open("/project/train/src_repo/train.txt", 'w')
    val = open("/project/train/src_repo/val.txt", 'w')

    for i in files: //100
        num = randint(1, 100)
        if num < 95:
            train.write(i + '\n')
        else:
            val.write(i + '\n')
    train.close()
    val.close()

