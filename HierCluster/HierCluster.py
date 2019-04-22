'''implement hierarchical cluster algorithm by ksy'''
# -*- coding:utf-8 -*-
import math
import pickle as pk
from codecs import open as cd_open
import matplotlib.pyplot as plt
import matplotlib.colors as cor


# define the information of every cluster
class CluInfo:
    def __init__(self, clu, cen):
        self.clu_index = clu    # the index of the cluster
        self.center = cen       # the center of the cluster

    # calculate the distance of two nodes(center vector of the cluster, Euclid Distance)
    def calculate_distance(self, cl_cen):
        dist = 0
        for i in range(0, len(cl_cen)):
            dist += math.pow((self.center[i]-cl_cen[i]), 2)
        return math.sqrt(dist)

    # merge two cluster and update the center
    def merge_two_cluster(self, clu):
        self.clu_index += clu.clu_index
        for i in range(0, len(self.center)):
            self.center[i] = (self.center[i] + clu.center[i])/2.0


# pre-process class
class ProData:
    def __init__(self):
        self.type_num = 0        # the type of the set with no label
        self.feature_num = 0     # the feature num of the sample
        self.sample_num = 0      # the num of all sample
        self.file_name = None    # the path of data set (the label pos for test accuracy)
        self.label_pos = 0       # label pos (None in reality, kept for test)
        self.type_size = []      # sample size by every type (kept for test)
        self.clust = []          # the cluster of the sample
        self.x_pos = 0           # the attribute index chosen to draw (x coordinate)
        self.y_pos = 0           # the attribute index chosen to draw (y coordinate)
        self.feature_max = []    # the maximum value by feature
        self.feature_min = []    # the minimum value by feature
        self.feature_name = []   # obtain the feature name list
        self.all_sample = None   # the sample with label

    # read config file and get parameter
    def readConfig(self):
        f = cd_open("config.cfg", 'r', encoding="utf-8")
        conf = f.readlines()
        temp = conf[0].split(',')  # the path of the data set
        self.file_name = temp[0]
        self.label_pos = int(temp[1])
        temp = conf[1].split(',')  # obtain type num, feature num, x/y coordinate, sample num
        self.type_num = int(temp[0])
        self.feature_num = int(temp[1])
        self.x_pos = int(temp[2])
        self.y_pos = int(temp[3])
        self.sample_num = int(temp[4])
        temp = conf[2].split(',')   # obtain the sample size every type name
        for i in range(0, self.type_num):
            self.type_size.append(int(temp[i]))
        temp1 = conf[3].split(',')  # obtain the minimum value every feature
        temp2 = conf[4].split(',')  # obtain the maximum value every feature
        temp3 = conf[5].split(',')  # obtain the feature name list
        for i in range(0, self.feature_num):
            self.feature_min.append(temp1[i])
            self.feature_max.append(temp2[i])
            self.feature_name.append(temp3[i])
        f.close()

    # write list in pkl format
    def writePkl(self, filename, lst):
        fout = open(filename, 'wb')  # pkl文件要采用流写入方式
        pk.dump(lst, fout)
        fout.close()

    # read pkl file in list
    def readPkl(self, filename):
        fin = open(filename, 'rb')
        lst = pk.load(fin)
        fin.close()
        return lst

    # process data set and get distance matrix (pre-process)
    def first_preProcess(self):
        try:
            self.readConfig()
            f = open(self.file_name, 'r')  # read sample in cache
            self.all_sample = f.readlines()
            f.close()
            start = 0
            for i in range(0, self.type_num):
                for j in range(start, start + self.type_size[i]):
                    if self.label_pos == 0:   # discard the label
                        temp = self.all_sample[j][:-1].split(',')
                        temp = temp[1:self.feature_num+1]
                    else:
                        temp = self.all_sample[j].split(',')
                        temp = temp[:self.feature_num]  # store the data after processing
                    temp = list(map(eval, temp))  # need to convert the string type to float
                    self.clust.append(CluInfo([j], temp))
                    temp.append(i)
                    self.all_sample[j] = temp
                start += self.type_size[i]
            tps = self.type_size + [self.x_pos, self.y_pos]
            self.all_sample = self.all_sample[0: self.sample_num]
            self.feature_min = list(map(eval, self.feature_min))
            self.feature_max = list(map(eval, self.feature_max))
            self.writePkl("prod_data/sample.pkl", self.all_sample)
            self.writePkl("prod_data/cluster.pkl", self.clust)
            self.writePkl("prod_data/type.pkl",[tps, self.feature_name, self.feature_min, self.feature_max])
        except Exception as e:
            print(e)

    # when train/test set .pkl file exist, read directly
    def second_preProcess(self):
        self.clust = self.readPkl("prod_data/cluster.pkl")
        self.all_sample = self.readPkl("prod_data/sample.pkl")
        temp = self.readPkl("prod_data/type.pkl")
        self.x_pos = temp[0][-2]
        self.y_pos = temp[0][-1]
        self.type_size = temp[0][:-2]
        self.feature_name = temp[1]
        self.feature_min = temp[2]
        self.feature_max = temp[3]


# the core function of hierarchical cluster
# with the function of testing the performance of hierarchical cluster
class Cluster:
    def __init__(self, tps, clu):
        self.dist_mat = []         # the distance matrix of every cluster
        self.clust = clu           # the cluster of the sample
        self.type_num = len(tps)   # the type num of the sample
        self.type_size = tps       # the num of every type (list is null in real situation)

    # calculate the distance matrix of all cluster for the first time
    def init_distance_matrix(self):  # the initial length of the cluster means the sample num
        for i in range(0, len(self.clust)):
            temp = []
            for j in range(0, i):
                temp.append(self.clust[i].calculate_distance(self.clust[j].center))
            temp.append(0)
            self.dist_mat.append(temp)

    # find the index(x, y) with minimum distance (x<y)
    def find_minimum_distance(self):
        min_dist = 9999999
        x = 0
        y = 0              # the num of the cluster
        for i in range(0, len(self.clust)):
            for j in range(0, i):
              if self.dist_mat[i][j] < min_dist:
                  min_dist = self.dist_mat[i][j]
                  x = j
                  y = i
        return x, y

    # merge two cluster and update the distance matrix
    def merge_and_update(self, x, y):
        self.clust[x].merge_two_cluster(self.clust[y])
        for i in range(0, x):  # for the cluster index < x
            self.dist_mat[x][i] = self.clust[x].calculate_distance(self.clust[i].center)
        for i in range(x+1, len(self.clust)):  # for the cluster index > x
            self.dist_mat[i][x] = self.clust[x].calculate_distance(self.clust[i].center)
            if i >= y:   # delete the elements whose index is y (y col)
                del(self.dist_mat[i][y])
        del(self.dist_mat[y])     # delete the elements whose index is y (y row)
        del(self.clust[y])        # delete the cluster y

    # get the accuracy of every cluster
    def get_accuracy(self, all_sample, label):
        correct = 0
        for i in range(0, self.type_num):   # for every cluster
            lbc = [0 for i in range(0, self.type_num)]  # count the num by label
            temp = self.clust[i].clu_index
            for j in range(0, len(temp)):   # get the type
                lbc[int(all_sample[temp[j]][-1])]+= 1
            correct += max(lbc)      # assume the type is the label with maximum num
            label.append(lbc.index(max(lbc)))
        return float(correct)/float(len(all_sample))

    # execute the cluster algorithm
    def execute_cluster(self, all_sample, label):
        self.init_distance_matrix()  # while cluster num is greater than aim num
        while len(self.clust)> self.type_num:
            x,y = self.find_minimum_distance()
            self.merge_and_update(x, y)
        print ("Hierarchical Cluster Accuracy = ", self.get_accuracy(all_sample, label))


# define the class to complete draw function
class Draw:
    def __init__(self, x, y, name, mini, maxi):
        self.x_pos = x            # the index used to draw x scatter
        self.y_pos = y            # the index used to draw y scatter
        self.feature_name = name  # the name of features
        self.feature_min = mini   # the minimum value by feature
        self.feature_max = maxi   # the maximum value by feature

    # choose two feature to draw scatter
    def draw_original_scatter(self, subplot, type_size, all_sample):
        start = 0
        colors = list(cor.cnames.keys())     # get the color key in the form of list
        for i in range(0, len(type_size)):  # scatter with different type
            x_cord = []
            y_cord = []
            for j in range(start, start+type_size[i]):
                x_cord.append(all_sample[j][self.x_pos])
                y_cord.append(all_sample[j][self.y_pos])
            start += type_size[i]
            subplot.scatter(x=x_cord, y=y_cord, s=15, c=colors[i+10] , marker='o')
        subplot.set_xlim(self.feature_min[self.x_pos]-0.4, self.feature_max[self.x_pos]+1)
        subplot.set_ylim(self.feature_min[self.y_pos]-0.4, self.feature_max[self.y_pos]+1)
        subplot.set_title("Original Sample")
        subplot.set_xlabel(self.feature_name[self.x_pos])
        subplot.set_ylabel(self.feature_name[self.y_pos])

    # choose two feature to draw scatter after cluster
    def draw_clustered_scatter(self, subplot, clust, all_sample, label):
        colors = list(cor.cnames.keys())  # get the color key in the form of list
        for i in range(0, len(clust)):   # scatter with different type
            x_cord = []
            y_cord = []
            for index in clust[i].clu_index:      # for every cluster
                x_cord.append(all_sample[index][self.x_pos])
                y_cord.append(all_sample[index][self.y_pos])
            subplot.scatter(x_cord, y_cord, s=15, c=colors[label[i]+10], marker='o')
        subplot.set_xlim(self.feature_min[self.x_pos]-0.4,self.feature_max[self.x_pos]+1)
        subplot.set_ylim(self.feature_min[self.y_pos]-0.4,self.feature_max[self.y_pos]+1)
        subplot.set_title("Clustered Sample")
        subplot.set_xlabel(self.feature_name[self.x_pos])
        subplot.set_ylabel(self.feature_name[self.y_pos])

    # execute the function to draw the cluster
    def execute_draw(self, type_size, clust, all_sample, label):
        plt.figure(num=0, figsize=(10, 5))
        figure = plt.figure(0)      # get the present figure
        org_plot = figure.add_subplot(1, 2, 1)  # divide the image into (i*j) block，the sequence is k
        clu_plot = figure.add_subplot(1, 2, 2)
        self.draw_original_scatter(org_plot, type_size, all_sample)
        self.draw_clustered_scatter(clu_plot, clust, all_sample, label)
        plt.show()


if __name__ == "__main__":     # execute function
    pre = "RAW"
    prod = ProData()
    if pre == "RAW":  # raw data, need to be processed
        prod.first_preProcess()  # if it's the first time to use the data set
    else:
        prod.second_preProcess()  # if there exist the data sets processed
    clu = Cluster(prod.type_size, prod.clust)
    label = []
    clu.execute_cluster(prod.all_sample, label)   # the main function to complete cluster
    draw = Draw(prod.x_pos, prod.y_pos, prod.feature_name, prod.feature_min,prod.feature_max)
    draw.execute_draw(prod.type_size, clu.clust, prod.all_sample,label)