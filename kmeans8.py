import math
import random
import time
from tkinter import *
from scipy.spatial.distance import cosine, jaccard

######################################################################
# This section contains functions for loading CSV (comma separated values)
# files and convert them to a dataset of instances.
# Each instance is a tuple of attributes. The entire dataset is a list
# of tuples.
######################################################################


# Loads a CSV files into a list of tuples.
# Ignores the first row of the file (header).
# Numeric attributes are converted to floats, nominal attributes
# are represented with strings.
# Parameters:
#   fileName: name of the CSV file to be read
# Returns: a list of tuples
def loadCSV(fileName):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close()
    del lines[0]  # remove the header
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        dataset.append(instance)
    return dataset


# Converts a comma separated string into a tuple
# Parameters
#   line: a string
# Returns: a tuple
def lineToTuple(line):
    # remove leading/trailing witespace and newlines
    cleanLine = line.strip()
    # get rid of quotes
    cleanLine = cleanLine.replace('"', "")
    # separate the fields
    lineList = cleanLine.split(",")
    # convert strings into numbers
    stringsToNumbers(lineList)
    lineTuple = tuple(lineList)
    return lineTuple


# Destructively converts all the string elements representing numbers
# to floating point numbers.
# Parameters:
#   myList: a list of strings
# Returns None
def stringsToNumbers(myList):
    for i in range(len(myList)):
        if isValidNumberString(myList[i]):
            myList[i] = float(myList[i])


# Checks if a given string can be safely converted into a positive float.
# Parameters:
#   s: the string to be checked
# Returns: True if the string represents a positive float, False otherwise
def isValidNumberString(s):
    if len(s) == 0:
        return False
    if len(s) > 1 and s[0] == "-":
        s = s[1:]
    for c in s:
        if c not in "0123456789.":
            return False
    return True


######################################################################
# This section contains functions for clustering a dataset
# using the k-means algorithm.
######################################################################

# def cosine_distance(instance1, instance2):
#     dot_product = sum(a * b for a, b in zip(instance1[1:], instance2[1:]))
#     magnitude1 = math.sqrt(sum(a**2 for a in instance1[1:]))
#     magnitude2 = math.sqrt(sum(a**2 for a in instance2[1:]))
#     if magnitude1 == 0 or magnitude2 == 0:
#         return 1  # Max distance if magnitude is zero
#     return 1 - dot_product / (magnitude1 * magnitude2)

# def jaccard_distance(instance1, instance2):
#     set1 = set(instance1[1:])
#     set2 = set(instance2[1:])
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
#     if union == 0:
#         return 1  # Max distance if union is zero
#     return 1 - intersection / union


def cosine_distance(instance1, instance2):
    vector1 = instance1[1:]
    vector2 = instance2[1:]
    return cosine(vector1, vector2)


def jaccard_distance(instance1, instance2):
    vector1 = instance1[1:]
    vector2 = instance2[1:]
    return jaccard(vector1, vector2)


def distance(instance1, instance2, metric="euclidean"):
    if instance1 is None or instance2 is None:
        return float("inf")
    if metric == "euclidean":
        return sum((a - b) ** 2 for a, b in zip(instance1[1:], instance2[1:]))
    elif metric == "cosine":
        return cosine_distance(instance1, instance2)
    elif metric == "jaccard":
        return jaccard_distance(instance1, instance2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    if numInstances == 0:
        return
    numAttributes = len(instanceList[0])
    means = [name] + [0] * (numAttributes - 1)
    for instance in instanceList:
        for i in range(1, numAttributes):
            means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    return tuple(means)


def assign(instance, centroids, metric):
    minDistance = distance(instance, centroids[0], metric)
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(instance, centroids[i], metric)
        if d < minDistance:
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex


def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList


def assignAll(instances, centroids, metric):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids, metric)
        clusters[clusterIndex].append(instance)
    return clusters


def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "centroid" + str(i)
        centroid = meanInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids


def kmeans(
    instances,
    k,
    metric="euclidean",
    animation=False,
    initCentroids=None,
    max_iterations=100,
    stop_type="centroid",
):
    start_time = time.time()
    result = {}
    if initCentroids == None or len(initCentroids) < k:
        # randomly select k initial centroids
        random.seed(time.time())
        centroids = random.sample(instances, k)
    else:
        centroids = initCentroids
    prevCentroids = []
    prev_sse = float("inf")
    if animation:
        delay = 1.0  # seconds
        canvas = prepareWindow(instances)
        clusters = createEmptyListOfLists(k)
        clusters[0] = instances
        paintClusters2D(canvas, clusters, centroids, "Initial centroids")
        time.sleep(delay)
    iteration = 0
    # while centroids != prevCentroids:

    while True:
        iteration += 1
        clusters = assignAll(instances, centroids, metric)
        if animation:
            paintClusters2D(canvas, clusters, centroids, "Assign %d" % iteration)
            time.sleep(delay)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)

        withinss = computeWithinss(clusters, centroids, metric=metric)
        if animation:
            paintClusters2D(
                canvas,
                clusters,
                centroids,
                "Update %d, withinss %.1f" % (iteration, withinss),
            )
            time.sleep(delay)

        if check_stop_conditions(
            iteration,
            max_iterations,
            prevCentroids,
            centroids,
            prev_sse,
            withinss,
            type=stop_type,
        ):
            break

        prev_sse = withinss

    end_time = time.time()
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["iterations"] = iteration
    result["time_taken"] = end_time - start_time
    return result


def check_stop_conditions(
    iteration,
    max_iterations,
    prevCentroids,
    centroids,
    prev_sse,
    current_sse,
    type="centroid",
):
    if type == "iteration":
        return iteration >= max_iterations
    elif type == "centroid":
        return prevCentroids == centroids

    elif type == "sse":
        
        if prev_sse is None or current_sse is None:
            return False
        return current_sse >= prev_sse
    else:
        raise ValueError(f"Invalid stop condition type: {type}")


def computeWithinss(clusters, centroids, metric):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            result += distance(centroid, instance, metric)
    return result


# Repeats k-means clustering n times, and returns the clustering
# with the smallest withinss
def repeatedKMeans(instances, k, n):
    bestClustering = {}
    bestClustering["withinss"] = float("inf")
    for i in range(1, n + 1):
        print(
            "k-means trial %d," % i,
        )
        trialClustering = kmeans(instances, k)
        print("withinss: %.1f" % trialClustering["withinss"])
        if trialClustering["withinss"] < bestClustering["withinss"]:
            bestClustering = trialClustering
            minWithinssTrial = i
    print("Trial with minimum withinss:", minWithinssTrial)
    return bestClustering


######################################################################
# This section contains functions for visualizing datasets and
# clustered datasets.
######################################################################


def printTable(instances):
    for instance in instances:
        if instance != None:
            line = instance[0] + "\t"
            for i in range(1, len(instance)):
                line += "%.2f " % instance[i]
            print(line)


def extractAttribute(instances, index):
    result = []
    for instance in instances:
        result.append(instance[index])
    return result


def paintCircle(canvas, xc, yc, r, color):
    canvas.create_oval(xc - r, yc - r, xc + r, yc + r, outline=color)


def paintSquare(canvas, xc, yc, r, color):
    canvas.create_rectangle(xc - r, yc - r, xc + r, yc + r, fill=color)


def drawPoints(canvas, instances, color, shape):
    random.seed(0)
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2 * margin) / (maxX - minX)
    scaleY = float(height - 2 * margin) / (maxY - minY)
    for instance in instances:
        x = 5 * (random.random() - 0.5) + margin + (instance[1] - minX) * scaleX
        y = (
            5 * (random.random() - 0.5)
            + height
            - margin
            - (instance[2] - minY) * scaleY
        )
        if shape == "square":
            paintSquare(canvas, x, y, 5, color)
        else:
            paintCircle(canvas, x, y, 5, color)
    canvas.update()


def connectPoints(canvas, instances1, instances2, color):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2 * margin) / (maxX - minX)
    scaleY = float(height - 2 * margin) / (maxY - minY)
    for p1 in instances1:
        for p2 in instances2:
            x1 = margin + (p1[1] - minX) * scaleX
            y1 = height - margin - (p1[2] - minY) * scaleY
            x2 = margin + (p2[1] - minX) * scaleX
            y2 = height - margin - (p2[2] - minY) * scaleY
            canvas.create_line(x1, y1, x2, y2, fill=color)
    canvas.update()


def mergeClusters(clusters):
    result = []
    for cluster in clusters:
        result.extend(cluster)
    return result


def prepareWindow(instances):
    width = 500
    height = 500
    margin = 50
    root = Tk()
    canvas = Canvas(root, width=width, height=height, background="white")
    canvas.pack()
    canvas.data = {}
    canvas.data["margin"] = margin
    setBounds2D(canvas, instances)
    paintAxes(canvas)
    canvas.update()
    return canvas


def setBounds2D(canvas, instances):
    attributeX = extractAttribute(instances, 1)
    attributeY = extractAttribute(instances, 2)
    canvas.data["minX"] = min(attributeX)
    canvas.data["minY"] = min(attributeY)
    canvas.data["maxX"] = max(attributeX)
    canvas.data["maxY"] = max(attributeY)


def paintAxes(canvas):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    canvas.create_line(
        margin / 2,
        height - margin / 2,
        width - 5,
        height - margin / 2,
        width=2,
        arrow=LAST,
    )
    canvas.create_text(margin, height - margin / 4, text=str(minX), font="Sans 11")
    canvas.create_text(
        width - margin, height - margin / 4, text=str(maxX), font="Sans 11"
    )
    canvas.create_line(
        margin / 2, height - margin / 2, margin / 2, 5, width=2, arrow=LAST
    )
    canvas.create_text(
        margin / 4, height - margin, text=str(minY), font="Sans 11", anchor=W
    )
    canvas.create_text(margin / 4, margin, text=str(maxY), font="Sans 11", anchor=W)
    canvas.update()


def showDataset2D(instances):
    canvas = prepareWindow(instances)
    paintDataset2D(canvas, instances)


def paintDataset2D(canvas, instances):
    canvas.delete(ALL)
    paintAxes(canvas)
    drawPoints(canvas, instances, "blue", "circle")
    canvas.update()


def showClusters2D(clusteringDictionary):
    clusters = clusteringDictionary["clusters"]
    centroids = clusteringDictionary["centroids"]
    withinss = clusteringDictionary["withinss"]
    canvas = prepareWindow(mergeClusters(clusters))
    paintClusters2D(canvas, clusters, centroids, "Withinss: %.1f" % withinss)


def paintClusters2D(canvas, clusters, centroids, title=""):
    canvas.delete(ALL)
    paintAxes(canvas)
    colors = ["blue", "red", "green", "brown", "purple", "orange"]
    for clusterIndex in range(len(clusters)):
        color = colors[clusterIndex % len(colors)]
        instances = clusters[clusterIndex]
        centroid = centroids[clusterIndex]
        drawPoints(canvas, instances, color, "circle")
        if centroid != None:
            drawPoints(canvas, [centroid], color, "square")
        connectPoints(canvas, [centroid], instances, color)
    width = canvas.winfo_reqwidth()
    canvas.create_text(width / 2, 20, text=title, font="Sans 14")
    canvas.update()


######################################################################
# Test code
######################################################################

# dataset = loadCSV("/Users/yanjiefu/Downloads/tshirts-G.csv")
# showDataset2D(dataset)
# clustering = kmeans(dataset, 3, True)
# printTable(clustering["centroids"])
