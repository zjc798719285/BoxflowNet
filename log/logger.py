import scipy.io as sio

class Logger(object):

    def __init__(self, path):
        self.path = path
        self.dict = {}

    def write(self, names, values):
        for name_i, value_i in zip(names, values):
            if name_i in self.dict:
                self.dict[name_i].append(value_i)
            else:
                self.dict[name_i] = []
                self.dict[name_i].append(value_i)
    def savetomat(self):
        sio.savemat(self.path, self.dict)