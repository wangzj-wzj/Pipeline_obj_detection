#coding=utf-8
# 获取文件夹中的文件路径
import os
def getFilePathList_2(dirPath, partOfFileName=''):
    filePath_list = []
    for walk in os.walk(dirPath):
        subDirPath = walk[0]
        fileName_list = [k for k in walk[2] if partOfFileName in k]
        filePath_list.extend([os.path.join(subDirPath, k) for k in fileName_list])
    return filePath_list

#    
import xml.etree.ElementTree as ET
def xml2csv(xmlFilePath_list, csvFilePath):
    className_set = set()
    with open(csvFilePath, 'w') as csvFile:     
        for xmlFilePath in xmlFilePath_list:
            xmlFileName = os.path.split(xmlFilePath)[1]
            imageFileName = xmlFileName[:-4] + '.jpg'
            xmlDirPath = os.path.split(xmlFilePath)[0]
            if 'VOC' in xmlFilePath:
                imageDirPath = xmlDirPath.replace('Annotations', 'JPEGImages')
            else:
                imageDirPath = xmlDirPath
            imageFilePath = os.path.join(imageDirPath, imageFileName)
            with open(xmlFilePath, 'r') as file:
                fileContent = file.read()
            root = ET.XML(fileContent)
            for object_item in root.iter('object'):
                className = object_item.find('name').text
                className_set.add(className)
                bndbox = object_item.find('bndbox')
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text
                box = xmin, ymin, xmax, ymax
                csvFile.write(imageFilePath + ',' + ','.join(box) + ',' + className)
                csvFile.write('\n')
    return className_set

# 解析运行代码文件时传入的参数
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirPath', type=str)
    parser.add_argument('-p', '--percent', type=float, default=0.05)
    parser.add_argument('-t', '--train', type=str, default='train.csv')
    parser.add_argument('-te', '--test', type=str, default='test.csv')
    parser.add_argument('-c', '--classes', type=str, default='class.csv')
    argument_namespace = parser.parse_args()
    return argument_namespace

# 主函数
import random  
from sklearn.preprocessing import LabelEncoder
if __name__ == "__main__":
    argument_namespace = parse_args()
    dirPath = argument_namespace.dirPath
    test_percent = argument_namespace.percent
    train_csvFilePath = argument_namespace.train
    test_csvFilePath = argument_namespace.test
    class_csvFilePath = argument_namespace.classes
    xmlFilePath_list = getFilePathList_2(dirPath, '.xml')
    test_xmlFilePath_list = random.sample(xmlFilePath_list,
        int(len(xmlFilePath_list) * test_percent))
    train_xmlFilePath_list = list(set(xmlFilePath_list) - set(test_xmlFilePath_list))
    train_className_set = xml2csv(train_xmlFilePath_list, train_csvFilePath)
    test_className_set = xml2csv(test_xmlFilePath_list, test_csvFilePath)
    className_set = train_className_set | test_className_set
    labelEncoder = LabelEncoder()
    labelEncoder.fit(list(className_set))
    with open(class_csvFilePath, 'w') as file:
        for index, className in enumerate(labelEncoder.classes_):
            file.write(className + ',' + str(index) + '\n')

