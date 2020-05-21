import caffe
import numpy as np
import cv2
import numpy.random as random

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.imgdir = ""
        self.maskdir = ""
        self.imgtxt = "/home/yshi10/unet/imglist.txt"
        self.masktxt = "/home/yshi10/unet/masklist.txt"
        self.random = True
        self.seed = None

        if len(top) != 2:
            raise Exception("Need to define two tops: data and mask.")

        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.lines = open(self.imgtxt, 'r').readlines()
        self.lines1 = open(self.masktxt, 'r').readlines()
        self.idx = 0

        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.lines) - 1)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.idx)
        self.mask = self.load_mask(self.idx)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.mask.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.mask

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.lines) - 1)
        else:
            self.idx += 1
            if self.idx == len(self.lines):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):

        imname = self.imgdir + self.lines[idx]
        imname = imname[:-1]
        print ('load img %s' %imname)
        im = cv2.imread(imname)
        #im = cv2.imread(imname)
        #print im.shape
        im = cv2.resize(im,(572,572))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = np.array(im, np.float64)
        im /= 255.0
        im -= 0.5
        return im[np.newaxis, :]

    def load_mask(self, idx):
        outimg = np.empty((2,572,572))
        imname = self.maskdir + self.lines1[idx]
        imname = imname[:-1]
        print ('load mask %s' %imname)
        im = cv2.imread(imname)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im,(572,572))
        ret, img = cv2.threshold(im, 0.5, 1.0, cv2.THRESH_BINARY)
	#ret, back = cv2.threshold(im, 0.5, 1.0, cv2.THRESH_BINARY_INV)
	#outimg[0, ...] = img;
	#outimg[1, ...] = back;
	#outimg.astype(np.uint8)
        return img[np.newaxis, :]
