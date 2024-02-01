import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
import pandas as pd

pixelClasses = {(115, 0, 108): 'GLD', (122, 1, 145): 'INF', (148, 47, 216): 'FOL', (242, 246, 254): 'HYP',
                (130, 9, 181): 'RET', (157, 85, 236): 'PAP', (106, 0, 73): 'EPI', (168, 123, 248): 'KER',
                (0, 0, 0): 'BKG', (255, 255, 127): 'BCC', (142, 255, 127): 'SCC', (127, 127, 255): 'IEC'}
classToPixel = {cls: pixel for pixel, cls in pixelClasses.items()}
segmentAverages = {'GLD': [0, 0], 'INF': [0, 0], 'FOL': [0, 0], 'HYP': [0, 0], 'RET': [0, 0], 'PAP': [0, 0],
                   'EPI': [0, 0], 'KER': [0, 0], 'BKG': [0, 0], 'BCC': [0, 0], 'SCC': [0, 0], 'IEC': [0, 0]}
classPercentages = {'GLD': [0, 0, 0], 'INF': [0, 0, 0], 'FOL': [0, 0, 0], 'HYP': [0, 0, 0], 'RET': [0, 0, 0],
                    'PAP': [0, 0, 0], 'EPI': [0, 0, 0], 'KER': [0, 0, 0], 'BKG': [0, 0, 0], 'BCC': [0, 0, 0],
                    'SCC': [0, 0, 0], 'IEC': [0, 0, 0]}


def closestClass(pixel):
    minDist = math.inf
    closest_cls = None

    for cls_pixel, cls_name in pixelClasses.items():
        dist = math.sqrt(sum((p - c) ** 2 for p, c in zip(pixel, cls_pixel)))
        if dist < minDist:
            minDist = dist
            closest_cls = cls_name

    return closest_cls


def HSV(inputImg):
    hsv_image = cv2.cvtColor(inputImg, cv2.COLOR_RGB2HSV)

    # Split HSV channels
    hue, sat, val = cv2.split(hsv_image)

    return hue, sat, val


def sharpenImage(inputImg, sharpnessFactor=2):
    b, g, r = cv2.split(inputImg)

    # Apply Laplacian filter to each color channel
    laplacian_b = cv2.Laplacian(b, cv2.CV_64F) * sharpnessFactor
    laplacian_g = cv2.Laplacian(g, cv2.CV_64F) * sharpnessFactor
    laplacian_r = cv2.Laplacian(r, cv2.CV_64F) * sharpnessFactor

    # 64F --> INT16
    laplacian_b = np.clip(laplacian_b, 0, 255).astype(np.int16)
    laplacian_g = np.clip(laplacian_g, 0, 255).astype(np.int16)
    laplacian_r = np.clip(laplacian_r, 0, 255).astype(np.int16)

    # Convert the Laplacian results to appropriate data type and scale
    sharpened_b = cv2.convertScaleAbs(b - laplacian_b)
    sharpened_g = cv2.convertScaleAbs(g - laplacian_g)
    sharpened_r = cv2.convertScaleAbs(r - laplacian_r)

    # Merge the sharpened color channels back into a single image
    sharpImg = cv2.merge((sharpened_b, sharpened_g, sharpened_r))

    # Display the original and sharpened images
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(sharpImg, cv2.COLOR_BGR2RGB))
    # plt.title('Sharpened Image')
    #
    # plt.tight_layout()
    # plt.show()

    return sharpImg


def closestSegment(values):
    closest_classes = []
    for value in values:
        min_distance = np.inf
        closest_class = None
        for class_name, average_values in segmentAverages.items():
            distance_hue = abs(value[0] - average_values[0])
            distance_saturation = abs(value[1] - average_values[1])
            distance = min(distance_hue, distance_saturation)

            if distance < min_distance:
                min_distance = distance
                closest_class = class_name

        closest_classes.append(closest_class)
    return closest_classes


def checkUniqueness(lst):
    return len(lst) == len(set(lst))


rootPath = '/Users/maazinzaidi/Desktop/Queensland Dataset CE42'
rootDir = os.listdir(rootPath)
if '.DS_Store' in rootDir:
    rootDir.remove('.DS_Store')
print(rootDir)
imageCount = 0
allLabels = []
for i in range(len(rootDir)):
    currentFolder = os.listdir(os.path.join(rootPath, rootDir[i]))
    if '.DS_Store' in currentFolder:
        currentFolder.remove('.DS_Store')
    Images = os.listdir(os.path.join(rootPath, rootDir[i], 'Images'))
    if '.DS_Store' in Images:
        Images.remove('.DS_Store')

    imageCount += len(Images)

    Masks = os.listdir(os.path.join(rootPath, rootDir[i], 'Masks'))
    if '.DS_Store' in Masks:
        Masks.remove('.DS_Store')

allImages = np.zeros((imageCount, 256, 256, 3), dtype=np.uint8)
allMasks = np.zeros((imageCount, 256, 256, 3), dtype=np.uint8)

count = 0
for i in range(len(rootDir)):
    currentFolder = os.listdir(os.path.join(rootPath, rootDir[i]))
    if '.DS_Store' in currentFolder:
        currentFolder.remove('.DS_Store')
    Images = os.listdir(os.path.join(rootPath, rootDir[i], 'Images'))
    if '.DS_Store' in Images:
        Images.remove('.DS_Store')

    Masks = os.listdir(os.path.join(rootPath, rootDir[i], 'Masks'))
    if '.DS_Store' in Masks:
        Masks.remove('.DS_Store')

    for j in range(len(Images)):
        imagePath = os.path.join(rootPath, rootDir[i], 'Images', Images[j])
        currentImage = cv2.imread(imagePath, -1)
        allImages[count, :, :, :] = currentImage

        maskPath = os.path.join(rootPath, rootDir[i], 'Masks', Masks[j])
        currentMask = cv2.imread(maskPath, -1)
        allMasks[count, :, :, :] = currentMask
        allLabels.append(i)
        # print(i, os.path.join(rootPath, rootDir[i]))
        count += 1

# LABELS : 0 - BCC, 1 - SCC, 2 - IEC
classNames = ['BCC', 'SCC', 'IEC']
training = 1400
testing = 100

trainingImages, testingImages, trainingMasks, testingMasks, trainingLabels, testingLabels = train_test_split(
    allImages, allMasks, allLabels, train_size=training, test_size=testing, random_state=0)

labelCounter = [0, 0, 0]
# TRAINING
for m in range(1400):
    print(m)
    currentImg = trainingImages[m]
    currentMask = trainingMasks[m]

    sharpenedImg = sharpenImage(currentImg)
    imgH, imgS, imgV = HSV(sharpenedImg)
    grayImg = cv2.cvtColor(sharpenedImg, cv2.COLOR_BGR2GRAY)
    allColours = []

    for i in range(256):
        for j in range(256):
            colour = tuple(currentMask[i, j, :])
            if colour not in allColours:
                allColours.append(colour)

    for i in range(len(allColours)):
        colour = allColours[i]
        if colour not in pixelClasses:
            closest = closestClass(colour)
            allColours[i] = classToPixel[closest]

    uniqueColours = [x for i, x in enumerate(allColours) if x not in allColours[:i]]

    currentSegments = len(uniqueColours)
    segmentAvg = []
    uniqueSegments = [pixelClasses[uniqueColours[i]] for i in range(currentSegments)]

    for currentClass in uniqueColours:
        currentTotalH = 0
        currentTotalS = 0
        currentCount = 0
        for i in range(256):
            for j in range(256):
                if all(currentMask[i, j] == currentClass):
                    currentTotalH += imgH[i, j]
                    currentTotalS += imgS[i, j]
                    currentCount += 1

        if currentCount != 0:
            averageH = currentTotalH / currentCount
            averageS = currentTotalS / currentCount
            segmentAvg.append([averageH, averageS])
        else:
            segmentAvg.append([0, 0])

    for i in range(len(uniqueColours)):
        currentClass = pixelClasses[uniqueColours[i]]
        if segmentAverages[currentClass] == [0, 0]:
            segmentAverages[currentClass] = segmentAvg[i]
        else:
            segmentAverages[currentClass][0] = (segmentAverages[currentClass][0] + segmentAvg[i][0]) / 2
            segmentAverages[currentClass][1] = (segmentAverages[currentClass][1] + segmentAvg[i][1]) / 2

    currentLabel = trainingLabels[m]
    labelCounter[currentLabel] += 1

    for i in range(len(uniqueSegments)):
        classPercentages[uniqueSegments[i]][currentLabel] += 1


for segment, percentages in classPercentages.items():
    currentSum = sum(percentages)
    for i in range(len(percentages)):
        percentages[i] = round(((percentages[i] / currentSum) * 100), 2)


print(f"{'Index':<6} {'Class':<6} {'Percentage':<15}")
# Print values from each key-value pair
for index, (key, value) in enumerate(classPercentages.items()):
    print(f"{index + 1:<6} {key:<6} {value[0]:<5} {value[1]:<5} {value[2]:<5}")
print()

print(f"{'Index':<6} {'Class':<6} {'Average':<15}")
for index, (key, value) in enumerate(segmentAverages.items()):
    print(f"{index + 1:<4} Class: {key:<4} Average: {value}")
print()
print('\n\n\n\n\n\n\n\n\n\n')


allMaxClass = []
imageProbabilities = []
groundTruths = []
diceCoefficients = []
# TESTING
for m in range(3):
    print(m)
    myImg = testingImages[m]
    groundTruthImg = testingMasks[m]
    currentTLabel = testingLabels[m]
    sharpenedImg = sharpenImage(myImg)
    rows, cols, _ = myImg.shape
    h, s, v = HSV(sharpenedImg)
    h = h.flatten().reshape(-1, 1)
    s = s.flatten().reshape(-1, 1)
    v = v.flatten().reshape(-1, 1)
    featureVector = np.hstack((h, s))
    chosenSegments = []

    for i in range(12, 0, -1):
        # print(i)
        numClusters = i
        kmeansColour = KMeans(n_clusters=numClusters, random_state=0)
        kmeansColour.fit(featureVector)
        centers = kmeansColour.cluster_centers_

        chosenSegments = closestSegment(centers)
        # print(chosenSegments)

        boolFlag = checkUniqueness(chosenSegments)

        if boolFlag:
            finalNumClusters = i
            break

    # print("\nOut of Loop")
    # print(chosenSegments)

    labelsColour = kmeansColour.labels_
    segImgLabels = (labelsColour.reshape(cols, rows))
    segmentedImg = np.zeros((rows, cols, 3), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            currentLabel = segImgLabels[i, j]
            segmentedImg[i, j, :] = classToPixel[chosenSegments[currentLabel]]

    # cv2.imshow("Segmented Image", segmentedImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # filename = f"/Users/maazinzaidi/Dropbox/EME/EC-312 Digital Image Processing/Lab/Project/Sample Outputs/Mask{m+1}.jpg"
    # cv2.imwrite(filename, groundTruthImg)
    # filename2 = f"/Users/maazinzaidi/Dropbox/EME/EC-312 Digital Image Processing/Lab/Project/Sample Outputs/SI{m+1}.jpg"
    # cv2.imwrite(filename2, segmentedImg)

    numberOfSegments = len(chosenSegments)

    classProbability = [0, 0, 0]
    for i in range(numberOfSegments):
        currentSegment = chosenSegments[i]
        for j in range(len(classProbability)):
            classProbability[j] += classPercentages[currentSegment][j]

    classProbability = [round(classProbability[j] / numberOfSegments, 2) for j in range(len(classProbability))]

    # print("Probability of Likelihood:")
    # for i in range(3):
    #     print(f"{classNames[i]}: {classProbability[i]:.2f}%")


    maxProbIndex = classProbability.index(max(classProbability))
    maxClassName = classNames[maxProbIndex]
    # print(f"\nTherefore, it is most likely to be of type: {maxClassName}")


    # PREPARING FOR EXCEL FILE
    allMaxClass.append(maxClassName)
    imageProbabilities.append(classProbability)
    groundTruths.append(classNames[currentTLabel])


# SAVING TO EXCEL FILE
data = pd.DataFrame()
data['Predicted'] = allMaxClass
data['Ground Truth'] = groundTruths

# Save the DataFrame to an Excel file
data.to_excel('/Users/maazinzaidi/Dropbox/EME/EC-312 Digital Image Processing/Lab/Project/Results.xlsx', index=False)