import numpy as np
import random

def noCorrection(x, y):
    return [1]

def predictX(w, x, y):
    return x

def predictY(w, x, y):
    return y

# simple feature vector to predict x based on x and constant factor.
def featureVectorSimpleX(x, y):
    return [1, x]


def predictSimpleX(w, x, y):
    return x + w[0] + w[1] * x

# simple feature vector to predict y based on y and constant factor.
def featureVectorSimpleY(x, y):
    return [1, y]

def predictSimpleY(w, x, y):
    return y + w[0] + w[1] * y

# advanced feature vector to predict x, y based on x, y and constant factor.
def featureVectorAdvanced(x, y):
    return [1, x, y]

def predictAdvancedX(w, x, y):
    return x + w[0] + x*w[1] + y*w[2] 

def predictAdvancedY(w, x, y):
    return y + w[0] + x*w[1] + y*w[2]

# more advanced feature vector to predict x, y based on x, y, xy, x^2, y^2, and constant factor.
def featureVectorMoreAdvanced(x, y):
    return [1, x, y, x*x, y*y, x*y]

def predictMoreAdvancedX(w, x, y):
    return x + w[0] + x*w[1] + y*w[2] + x*x*w[3] + y*y*w[4] + x*y*w[5]

def predictMoreAdvancedY(w, x, y):
    return y + w[0] + x*w[1] + y*w[2] + x*x*w[3] + y*y*w[4] + x*y*w[5]

# medium advanced feature vector to predict x, y based on x, y, x^2, y^2, and constant factor.
def featureVectorMediumAdvanced(x, y):
    return [1, x, y, x*x, y*y]

def predictMediumAdvancedX(w, x, y):
    return x + w[0] + x*w[1] + y*w[2] + x*x*w[3] + y*y*w[4]

def predictMediumAdvancedY(w, x, y):
    return y + w[0] + x*w[1] + y*w[2] + x*x*w[3] + y*y*w[4]

def assembleFeatureMatrix(touchTargetSequence, featureVector):
    featureMatrix = np.zeros_like(featureVector(0, 0))
    for touchTarget in touchTargetSequence:
        touch = touchTarget[0]
        ent = featureVector(touch[0], touch[1])
        featureMatrix = np.vstack([featureMatrix, ent])
    return np.matrix(featureMatrix[1:])

def computeXOffset(touchTargetSequence):
    for touchTarget in touchTargetSequence:
	touch, target = touchTarget
	yield target[0] - touch[0]

def computeYOffset(touchTargetSequence):
    for touchTarget in touchTargetSequence:
        touch, target = touchTarget
        yield target[1] - touch[1]

def learnW(touchTargetSequence, featureVector, computeOffset):
    X = assembleFeatureMatrix(touchTargetSequence, featureVector)
    offsets = np.matrix(list(computeOffset(touchTargetSequence)))
    offsetsColumn = np.transpose(np.matrix(offsets))
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, offsetsColumn))
    return w

#touchTargetSequence = [[(0.9, 0.9), (1, 1)], [(0.4, 0.4), (0.5, 0.5)], [(0.3, 0.4), (0.5, 0.5)], [(0.2, 0.1), (0.4, 0.6)], [(0.4, 0.4), (0.5, 0.5)]]


touchTargetBySubject = {}

with open("courseworkdata.csv", "r") as csv:
    for line in csv.readlines()[1:]:
        line = line.rstrip("\n")
        line = line.split(",")
        touch = float(line[3]), float(line[4])
        target = float(line[1]), float(line[2])
        subject = line[5]
        if subject not in touchTargetBySubject:
            touchTargetBySubject[subject] = [(touch, target)]
        else:
            touchTargetBySubject[subject].append((touch, target))

# I'm estimating the key radius to be such that 10 keys can fit horizontally on the keyboard, which is standard for qwerty keyboard (qwertyuiop).
keyRadius = (1.0/10)/2

def kFoldCV(touchTargetSequence, k, featureVectorX, featureVectorY, predictX, predictY, globalTouchTargetSequence = None):
    def foldAllApart(buckets, r):
        for bucket in buckets[:r]:
            for touchTarget in bucket:
                yield touchTarget
        for bucket in buckets[r+1:]:
            for touchTarget in bucket:
                yield touchTarget
    random.shuffle(touchTargetSequence)
    buckets = [touchTargetSequence[i::k] for i in range(k)]
    score = 0
    for i in range(k):
        if globalTouchTargetSequence is None:
            learnSequence = list(foldAllApart(buckets, i))
        else:
            learnSequence = globalTouchTargetSequence
        validateSequence = buckets[i]
        x_w = learnW(learnSequence, featureVectorX, computeXOffset)
        y_w = learnW(learnSequence, featureVectorY, computeYOffset)
        for touchTarget in validateSequence:
            touch, target = touchTarget
            predictedX = predictX(x_w, touch[0], touch[1])
            predictedY = predictY(y_w, touch[0], touch[1])
            distance = (predictedX - target[0])**2 + (predictedY - target[1])**2
            if distance < keyRadius**2:
                score += 1
    return score

def allTouchTarget():
    for key in touchTargetBySubject:
        for touchTarget in touchTargetBySubject[key]:
            yield touchTarget

globalTouchTargetSequence = list(allTouchTarget())
 
for key in touchTargetBySubject:
     print "*"*30, key, "*"*30
     print "*"*15, "noCorrection"
     print kFoldCV(touchTargetBySubject[key], 10, noCorrection, noCorrection, predictX, predictY)

     print "*"*15, "simpleX, simpleY"
     print "personalised score"
     print kFoldCV(touchTargetBySubject[key], 10, featureVectorSimpleX, featureVectorSimpleY, predictSimpleX, predictSimpleY)
     print "non-personalised score"
     print kFoldCV(touchTargetBySubject[key], 10, featureVectorSimpleX, featureVectorSimpleY, predictSimpleX, predictSimpleY, globalTouchTargetSequence = globalTouchTargetSequence)

     print "*"*15, "advancedX, advancedY"
     print "personalised score"
     print kFoldCV(touchTargetBySubject[key], 10, featureVectorAdvanced, featureVectorAdvanced, predictAdvancedX, predictAdvancedY)
     print "non-personalised score"
     print kFoldCV(touchTargetBySubject[key], 10, featureVectorAdvanced, featureVectorAdvanced, predictAdvancedX, predictAdvancedY, globalTouchTargetSequence = globalTouchTargetSequence)

     print "*"*15, "moreAdvanced, moreAdvanced"
     print "personalised score"
     print kFoldCV(touchTargetBySubject[key], 10, featureVectorMoreAdvanced, featureVectorMoreAdvanced, predictMoreAdvancedX, predictMoreAdvancedY)
     print "non-personalised score"
     print kFoldCV(touchTargetBySubject[key], 10,  featureVectorMoreAdvanced, featureVectorMoreAdvanced, predictMoreAdvancedX, predictMoreAdvancedY, globalTouchTargetSequence = globalTouchTargetSequence)

     print "*"*15, "mediumAdvanced"
     print "personalised score"
     print kFoldCV(touchTargetBySubject[key], 10,  featureVectorMediumAdvanced, featureVectorMediumAdvanced, predictMediumAdvancedX, predictMediumAdvancedY)
     print "non-personalised score"
     print kFoldCV(touchTargetBySubject[key], 10,  featureVectorMediumAdvanced, featureVectorMediumAdvanced, predictMediumAdvancedX, predictMediumAdvancedY, globalTouchTargetSequence = globalTouchTargetSequence)
