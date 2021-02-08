import numpy as np

class Label(object):
    def __init__(self, label):
        data = label.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0] 
        self.truncation = data[1] 
        self.occlusion = int(data[2]) 
        self.alpha = data[3] 

        self.xmin = data[4]
        self.ymin = data[5]
        self.xmax = data[6]
        self.ymax = data[7]
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        self.h = data[8] 
        self.w = data[9]
        self.l = data[10]
        self.t = (data[11],data[12],data[13]) 
        self.ry = data[14]

    def __str__(self):
        return self.type + ' : [' + str(self.box2d[0]) + ', ' + str(self.box2d[1]) + ', ' + str(self.box2d[2]) + ', ' + str(self.box2d[3]) +']'

    def __repr__(self):
        return self.__str__()