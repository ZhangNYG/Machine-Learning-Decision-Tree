#Author：Xianjie Zhang
#Date ：2017/4/23 spring@DaLian
#Author Affiliation :School of Innovation and Entrepreneurship of Dalian University of Technology


#!/usr/bin/python
# -*- coding: UTF-8 -*-
from numpy import *
from os import listdir
import math
import copy 


#读取文件中数据转化成矩阵，分为训练矩阵和验证矩阵
def file2matrix(filename):
    fr = open(filename,"r")
    linesInFile = fr.readlines()  #读取文件并且按照列表保存
    numberOfLines = len(linesInFile) #返回行数
    returnMat = zeros([numberOfLines,3]) #准备一个numberOflines*3的零矩阵
    classLabelVector = []                   #准备一个标签列表
    index = 0
    for line in linesInFile:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    fr.close()    
    returnMatTrain = zeros([int(numberOfLines*0.9),3])
    classLabelTrain = classLabelVector[0:int(numberOfLines*0.9)]
    classLabelValidation = classLabelVector[int(numberOfLines*0.9):]
    returnMatTrain = returnMat[0:int(numberOfLines*0.9),0:3]
    returnMatValidation = returnMat[int(numberOfLines*0.9):,0:3]
    return returnMat,classLabelVector,returnMatTrain,returnMatValidation,classLabelTrain,classLabelValidation

#数据归一化处理    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

#将归一化数据,以0.5为界限，将大于等于化为1，小于为0，并将类别标签一起组合起来
def assmbleMat(dataSet,dataLabel):
    numberOfLines = len(dataSet)
    returnAssMat = zeros([numberOfLines,4])
    for line in range(0,numberOfLines):
        returnAssMat[line,0:3] = dataSet[line,:]
        returnAssMat[line,3] = int(dataLabel[line])
        
    for line0 in range(0,numberOfLines):
        if returnAssMat[line0,0] >= 0.5:
            returnAssMat[line0,0] = 1
        else:
            returnAssMat[line0,0] = 0
            
    for line1 in range(0,numberOfLines):
        if returnAssMat[line1,1] >= 0.5:
            returnAssMat[line1,1] = 1
        else:
            returnAssMat[line1,1] = 0

    for line2 in range(0,numberOfLines):
        if returnAssMat[line2,2] >= 0.5:
            returnAssMat[line2,2] = 1
        else:
            returnAssMat[line2,2] = 0
            
    return returnAssMat
    
    

#计算信息增益
def calcEnt(dataSet):            
    numEntries = len(dataSet)   # 计算数据中的实例总数
    labelCounts = {}    # 创建字典，保存各类标签的数量
    for featVec in dataSet:    #计算各个类的数量the the number of unique elements and their occurance
        currentLabel = int(featVec[-1])
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        else:
            labelCounts[currentLabel] += 1
    ent = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
       # print (prob)
        ent -= prob * math.log(prob,2) 
    return ent

#划分数据集 value值应该要变化 当value==1，大于0.5 当value==0，小于0.5
 #attr为第几个属性，value是属性的性质
def splitDataSet(dataSet, attr, value):
    retDataSet = []
    #print ('splitDataSet读入的dataSet：\n',dataSet)
    #print ('dataSet类型：',type(dataSet))
    #print ('\n','attr,value:',attr,value)
    for featVec in dataSet:
        if featVec[attr] == value:
            
            reducedFeatVec = featVec[:attr]     #chop out axis used for splitting           
            list_reducedFeatVec = list(reducedFeatVec)
            list_featVec = list(featVec)
            #reducedFeatVec = concatenate(reducedFeatVec,featVec[attr+1:])
            list_reducedFeatVec.extend(list_featVec[attr+1:])
            reducedFeatVec = array(list_reducedFeatVec)
            retDataSet.append(list_reducedFeatVec)
    retDataSet = array(retDataSet)            
    return retDataSet

#计算信息熵，对所有属性划分数据集，找到最好的属性并且返回
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #属性数量the last column is used for the labels
    baseEntropy = calcEnt(dataSet)		#计算整个数据集的信息熵Ent(D)
    print (baseEntropy)
    bestInfoGain = 0.0; bestFeature = -1	
    for i in range(numFeatures):        #循环执行全部属性iterate over all the features
        featList = [example[i] for example in dataSet] #对每一列数据（属性）创建一个列表create a list of all the examples of this feature
        #print (featList)
        uniqueVals = set(featList)       #获得该属性的所有可能值set(‘abbba’)={‘a’, ’b’}get a set of unique values
        print ('所有可能属性：',uniqueVals)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)   #划分数据集，对于i==value划分
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        print ('信息增益：',infoGain)
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            #print (bestInfoGain)
            bestFeature = i
    return bestFeature                      #返回最好属性值returns an integer

#构建决策树
#当遍历完所有的属性，依然没有找到叶子节点，这时候就要用投票法确定叶子节点
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda itemLabel:itemLabel[1], reverse=True)
    print ('投票法确定叶子节点:',sortedClassCount[0][0])
    return sortedClassCount[0][0]

#构建决策树核心
def createTree(dataSet,attributes):
    classList = [example[-1] for example in dataSet]   #为最后列(label)创建列表
    if classList.count(classList[0]) == len(classList):  # 如果所有数据类都相同
        return classList[0]    #stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:   # 如果当前数据集中，这是最后一个待处理的属性stop splitting when there are no more features in dataSet
        return majorityCnt(classList)  # 采用投票法决定叶节点
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 判断哪个属性是最好的
    bestFeatLabel = attributes[bestFeat]  
    myTree = {bestFeatLabel:{}}  # 创建一个字典，保存决策树
    print ('\n\n\n\n\n' ,myTree, '\n\n\n\n\n\n')
    del(attributes[bestFeat])    # 删除已处理的属性
    featValues = [example[bestFeat] for example in dataSet] # 为属性创建列表
    uniqueVals = set(featValues)  # 获得该属性的所有可能取值(确定树节点的分支)
    for value in uniqueVals:
        subAttributes = attributes[:]       # 复制所有属性(除已删除的bestFeat)copy all of attributes, so trees don't mess up existing attributes
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subAttributes) # 递归创建树的节点(对每一个分支)
    return myTree

#测试决策树 一组数据
def classify(inputTree,featAttributes,testVec):
    firstStr = list(inputTree.keys())[0]   #获取当前决策树根节点属性
    secondDict = inputTree[firstStr]      #获取该节点下的剩余决策树
    featIndex = featAttributes.index(firstStr)  #根结点属性所处位置
    key = testVec[featIndex]                   #获取测试数据中该属性的值
    valueOfFeat = secondDict[key]             #找到该属性值的分支，并返回
    if isinstance(valueOfFeat, dict):       #如果该分支是字典，需要继续遍历
        classLabel = classify(valueOfFeat, featAttributes, testVec)
    else: classLabel = valueOfFeat        #否则，返回叶节点值
    return classLabel


#测试决策树正确率  全部数据
def testing(myTree,data_test,labels):  
    error=0.0  
    for i in range(len(data_test)):  
        if classify(myTree,labels,data_test[i])!=data_test[i][-1]:  
            error+=1  
    #print 'myTree %d' %error
    print ('\ntesting error:\n',error,'\n')
    return float(error)


#测试投票节点正确率
def testingMajor(major,data_test):  
    error=0.0  
    for i in range(len(data_test)):  
        if major!=data_test[i][-1]:  
            error+=1  
    #print 'major %d' %error
    print ('\ntestingMajor error:\n',error,'\n')
    return float(error)


#后剪枝  
def postPruningTree(inputTree,dataSet,data_test,labels):  
    firstStr = list(inputTree.keys())[0]               #获取当前决策树根节点属性
    secondDict=inputTree[firstStr]                     #获取该节点下的剩余决策树
    classList=[example[-1]
               for example in dataSet]     #classList 是所有的类别标签
    featkey = copy.deepcopy(firstStr)  
    labelIndex=labels.index(featkey)                   #lables 就是attributes = ['每年飞行里程','玩游戏时间','冰激凌消耗量']，labelIndex为在属性中的位置
    temp_labels = copy.deepcopy(labels)  
    del(labels[labelIndex])  
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict': 
            inputTree[firstStr][key] = postPruningTree(secondDict[key],splitDataSet(dataSet,labelIndex,key),splitDataSet(data_test,labelIndex,key),copy.deepcopy(labels))  
    if testing(copy.deepcopy(inputTree),copy.deepcopy(data_test),copy.deepcopy(temp_labels)) <= testingMajor(majorityCnt(classList),copy.deepcopy(data_test)):  
        return inputTree  
    return majorityCnt(classList)

#保存决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == '__main__':   #运行
    #读取数据返回矩阵
    returnMat,classLabelVector,returnMatTrain,returnMatValidation,classLabelTrain,classLabelValidation=file2matrix('datingTestSet2.txt')
    #   normDataSet, ranges, minVals =  autoNorm(returnMatTrain)
    #归一化处理
    normDataSetTrain, rangesTrain, minValsTrain =  autoNorm(returnMatTrain)
    #归一化后数据集成
    returnAssMax = assmbleMat(normDataSetTrain,classLabelTrain)   
    #print (returnAssMax)
    #print (len(returnAssMax))
    
    #归一化处理 验证集
    normDataSetValidation, rangesValidation, minValsValidation =  autoNorm(returnMatValidation)
    
    #归一化后数据集成
    returnAssMax_Validation = assmbleMat(normDataSetValidation,classLabelValidation)
    #print (returnAssMax_Validation)
    #print (len(returnAssMax_Validation))
    EntTrainOfD = calcEnt(returnAssMax)
    #print (EntTrainOfD)
    bestFeature = chooseBestFeatureToSplit(returnAssMax)
    #print ('最好属性标编号：',bestFeature)
    attributes = ['每年飞行里程','玩游戏时间','冰激凌消耗量']
    myTree =createTree(copy.deepcopy(returnAssMax),copy.deepcopy(attributes))
    print (myTree)
    
    post_tree = postPruningTree(myTree,copy.deepcopy(returnAssMax),copy.deepcopy(returnAssMax_Validation),copy.deepcopy(attributes))

    print ('后剪枝:\n',post_tree)
    #保存决策树
    storeTree(myTree,"txtOfmyTree.txt")
    #检查是否保存
    print ('保存的决策树为：')
    print (grabTree("txtOfmyTree.txt"))
    all_attributes = ['每年飞行里程','玩游戏时间','冰激凌消耗量']
    
    num_error = testing(myTree,returnAssMax_Validation,all_attributes)
    print ('num_error:',num_error)


