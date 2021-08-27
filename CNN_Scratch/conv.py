import numpy as np


class conv_3x3:

    #Convolution layer with three filters

    def __init__(self,num_filter):
        self.num_filter = num_filter
        self.filters = np.random.randn(num_filter, 3, 3) / 9


    def iterate(self,image):
        h,w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j


    def forward(self,input):
        h,w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filter))

        for im_region, i, j in self.iterate(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output