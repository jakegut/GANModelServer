import torch
import torchvision.utils as vutils
from torch.autograd import Variable

import json
import numpy
import models.dcgan as dcgan
#import matplotlib.pyplot as plt
import math

import random
from collections import OrderedDict

class Generator():

    out_height = 11
    out_width = 16
    batch_size = 1
    image_size = 32
    ngf = 64
    n_extra_layers = 0
    ngpu = 1

    def __init__(self, model_to_load, z_dims, nz):
        self.nz = nz
        self.z_dims = z_dims
        self.generator = dcgan.DCGAN_G(self.image_size, nz, z_dims, self.ngf, self.ngpu, self.n_extra_layers)
        deprecatedModel = torch.load(model_to_load, map_location=lambda storage, loc: storage)

        fixedModel = OrderedDict()
        for (goodKey,ignore) in self.generator.state_dict().items():
            # Take the good key and replace the : with . in order to get the deprecated key so the associated value can be retrieved
            badKey = goodKey.replace(":",".")
            #print(goodKey)
            #print(badKey)
        # Some parameter settings of the generator.state_dict() are not actually part of the saved models
        if badKey in deprecatedModel:
            goodValue = deprecatedModel[badKey]
            fixedModel[goodKey] = goodValue

        if not fixedModel:
            #print("LOAD REGULAR")
            #print(deprecatedModel)
            # If the fixedModel was empty, then the model was trained with the new labels, and the regular load process is fine
            self.generator.load_state_dict(deprecatedModel)
        else:
            # Load the parameters with the fixed labels  
            self.generator.load_state_dict(fixedModel)

    def generate(self, vector=None):
        random_latent = []
        if vector:
            random_latent = vector
        else:
            random_latent = [random.uniform(-1, 1) for _ in range(self.nz)]
        latent_vector = torch.FloatTensor(random_latent).view(self.batch_size, self.nz, 1, 1) 
        levels = self.generator(Variable(latent_vector, volatile=True))

        #levels.data = levels.data[:,:,:14,:28] #Cut of rest to fit the 14x28 tile dimensions

        level = levels.data.cpu().numpy()
        level = level[:,:,:self.out_height,:self.out_width] #Cut of rest to fit the 14x28 tile dimensions
        level = numpy.argmax( level, axis = 1)

        # Jacob: Only output first level, since we are only really evaluating one at a time
        return level[0].tolist()



def generate(modelToLoad, z_dims, nz, vector=None):
    out_height = 11
    out_width = 16
    batchSize = 1
	#nz = 10 #Dimensionality of latent vector

    imageSize = 32
    ngf = 64
    ngpu = 1
    n_extra_layers = 0

    generator = dcgan.DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers)
	#print(generator.state_dict()) 
	# This is a state dictionary that might have deprecated key labels/names
    deprecatedModel = torch.load(modelToLoad, map_location=lambda storage, loc: storage)

    fixedModel = OrderedDict()
    for (goodKey,ignore) in generator.state_dict().items():
        # Take the good key and replace the : with . in order to get the deprecated key so the associated value can be retrieved
        badKey = goodKey.replace(":",".")
        #print(goodKey)
        #print(badKey)
    # Some parameter settings of the generator.state_dict() are not actually part of the saved models
    if badKey in deprecatedModel:
        goodValue = deprecatedModel[badKey]
        fixedModel[goodKey] = goodValue

    if not fixedModel:
        #print("LOAD REGULAR")
        #print(deprecatedModel)
        # If the fixedModel was empty, then the model was trained with the new labels, and the regular load process is fine
        generator.load_state_dict(deprecatedModel)
    else:
        # Load the parameters with the fixed labels  
        generator.load_state_dict(fixedModel)

    random_latent = []
    if vector:
        random_latent = vector
    else:
        random_latent = [ random.uniform(-1.0, 1.0) ]*nz
    latent_vector = torch.FloatTensor(random_latent).view(batchSize, nz, 1, 1) 
    levels = generator(Variable(latent_vector, volatile=True))

    #levels.data = levels.data[:,:,:14,:28] #Cut of rest to fit the 14x28 tile dimensions

    level = levels.data.cpu().numpy()
    level = level[:,:,:out_height,:out_width] #Cut of rest to fit the 14x28 tile dimensions
    level = numpy.argmax( level, axis = 1)

    # Jacob: Only output first level, since we are only really evaluating one at a time
    return level[0].tolist()

    