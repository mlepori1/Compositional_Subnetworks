import os
import copy
import pickle
import numpy as np
# import matplotlib.pyplot as plt
#from PIL import Image

import itertools
import math

import cv2

from data_generation.shape import Shape
from data_generation.utils import *

"""
Defining all new tasks to probe for compositional subnetworks here.
"""

def sn_task_contact_inside(condition='c', odd_one_out = "all"):
    """
    The rule is that 3 images contain 2 objects, one of which is inside the other and in contact with the other

    odd_one_out determines which type of rule breaking is occuring in the odd one out.
    There are three types of breaking that can occur:
        (no_contact) Inner object is inside but not in contact with enclosing object
        (no_inside) Two objects are in contact, but one is not enclosing the other
        (no_inside_contact) Two objects are neither in contact nor enclosing one another
    Option "all" randomly samples these instances for generating the odd one out
    """
    n_samples = 4

    max_size = 0.5
    min_size = max_size/2

    size_a = np.random.rand(n_samples) * (max_size - min_size) + min_size
    size_b = np.random.rand(n_samples) * (size_a/2.5 - size_a/4) + size_a/4

    done = False
    max_attempts = 100


    range_1 = 1 - size_a[:,None]
    starting_1 = size_a[:,None]/2

    xy1 = np.random.rand(n_samples,2) * range_1 + starting_1

    xy2 = []
    shapes = []

    for i in range(n_samples-1):
        done = False

        s1 = Shape(gap_max=0.07, hole_radius=0.2)
        s2 = Shape(gap_max=0.01)
        for _ in range(max_attempts):

            sample = sample_contact_insideness(s1, s2, size_b[i]/size_a[i], a=None)
            if sample is not None:
                done = True

            if done:
                break
            else:
                s1.randomize()
                s2.randomize()

        xy2.append(sample)
        shapes.append([s1, s2])

    
    # Define which way we will break the rule to create the odd one out
    if odd_one_out == "all":
        odd_one_out = np.random.choice(["no_contact", "no_inside", "no_contact_inside"])

    # Sample new shapes of the same form
    s1 = Shape(gap_max=0.07, hole_radius=0.2)
    s2 = Shape(gap_max=0.01)

    if odd_one_out == "no_contact_inside":
        # The opposite of the normal inside function is no_contact_inside
        range_2 = 1 - size_b[-1]
        starting_2 = size_b[-1]/2
        xy2_odd = np.random.rand(100,2) * range_2 + starting_2
        xy1_odd = xy1[-1:]

        xy2_odd = xy2_odd[(np.abs(xy2_odd - xy1_odd) > (size_a[-1] + size_b[-1])/2).any(1)]
        xy2_odd = xy2_odd[0:1]

        shapes.append([s1,s2])
        xy2 = np.concatenate([np.array(xy2)*size_a[:-1,None] + xy1[:-1],xy2_odd], 0)

    elif odd_one_out == "no_contact":
        # this is just the normal insideness function
        done = False

        for _ in range(max_attempts):
            # In sample position inside many, the scale array is used to determine bounding boxes
            # around objects where objects shouldnt be sampled from. I artificially increase this
            # to prevent any inner objects from touching the outer object.
            # The function already does this, but it operates in a continuous x,y coordinate space,
            # which may result in objects touching once things are aliased in image pixels.
            # This hack ensures that you never run into this edge case.
            samples = sample_position_inside_many(s1, [s2], [(size_b[-1]/size_a[-1]) + .2])
            if len(samples)>0:
                done = True

            if done:
                break
            else:
                s1.randomize()
                s2.randomize()

        if not done:
            return np.zeros([100,100])

        xy2_odd = samples[0]
        xy2_odd = np.array(xy2_odd)*size_a[-1, None] + xy1[-1]
        xy2 = np.concatenate([np.array(xy2)*size_a[:-1,None] + xy1[:-1], xy2_odd.reshape(1, 2)], 0)
        shapes.append([s1, s2])

    elif odd_one_out == "no_inside":
        # This is just the normal contact function
        shape_sizes = [size_a[-1], size_b[-1]]
        positions, clump_size = sample_contact_many([s1, s2], shape_sizes) # Positions get you the offsets for both object 1 and object 2

        xy1_odd_init = np.random.rand(2) * (1-clump_size) + clump_size/2 # Need to resample the initial position of object 1 now that you                                                              
                                                                    # know the clump size
        
        # Apply offsets
        xy2_odd = positions[-1, :] + xy1_odd_init 
        xy1_odd = positions[0, :] + xy1_odd_init


        '''
        print(xy1.shape)
        xy1_odd = np.random.rand(2) * (1-clump_size) + clump_size/2 # Need to resample the position of object 1 now that you                                                              
                                                                    # know the clump size

        xy2_odd_relative_position = positions[-1, :] # positions are the relative positions of the other shapes in the clump
        xy2_odd = xy2_odd_relative_position + xy1_odd 
        '''
        xy1[-1] = xy1_odd

        shapes.append([s1, s2])
        #scale_and_shift = np.array(xy2)*size_a[:-1,None] + xy1[:-1]
        xy2 = np.concatenate([np.array(xy2)*size_a[:-1,None] + xy1[:-1],xy2_odd.reshape(1, 2)], 0)    
 
    xy2 = np.stack(xy2, axis=0)

    xy = np.stack([xy1, xy2], axis=1)
    size = np.stack([size_a, size_b], axis=1)
    if 'c' in condition:
        color = sample_random_colors(n_samples * 2)
        color1 = [np.ones([1, 3]) * color[i:i+1] for i in range(n_samples)]
        color2 = [np.ones([1, 3]) * color[i:i+1] for i in range(n_samples, 2*n_samples)]
        color = np.concatenate([color1, color2], axis=1)
    else:
        color = sample_random_colors(2)
        color1 = [np.ones([1, 3]) * color[0] for _ in range(n_samples)]
        color2 = [np.ones([1, 3]) * color[1] for _ in range(n_samples)]
        color = np.concatenate([color1, color2], axis=1)

        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shapes, color


# inside
# The images contain an object inside another object.
def sn_task_inside(condition='c'):

    n_samples = 4

    max_size = 0.5
    min_size = max_size/2

    size_a = np.random.rand(n_samples) * (max_size - min_size) + min_size
    size_b = np.random.rand(n_samples) * (size_a/2.5 - size_a/4) + size_a/4

    done = False
    max_attempts = 100


    range_1 = 1 - size_a[:,None]
    starting_1 = size_a[:,None]/2

    xy1 = np.random.rand(n_samples,2) * range_1 + starting_1

    xy2 = []
    shapes = []

    for i in range(n_samples-1):
        done = False

        s1 = Shape(gap_max=0.07, hole_radius=0.2)
        s2 = Shape(gap_max=0.01)
        for _ in range(max_attempts):
            # In sample position inside, the scale array is used to determine bounding boxes
            # around objects where objects shouldnt be sampled from. I artificially increase this
            # to prevent any inner objects from touching the outer object.
            # The function already does this, but it operates in a continuous x,y coordinate space,
            # which may result in objects touching once things are aliased in image pixels.
            # This hack ensures that you never run into this edge case.
            samples = sample_position_inside_1(s1, s2, size_b[i]/size_a[i] + .2)
            if len(samples)>0:
                done = True

            if done:
                break
            else:
                s1.randomize()
                s2.randomize()

        if not done:
            return np.zeros([100,100])

        xy2.append(samples[0])
        shapes.append([s1, s2])

    range_2 = 1 - size_b[-1]
    starting_2 = size_b[-1]/2
    xy2_odd = np.random.rand(100,2) * range_2 + starting_2
    xy1_odd = xy1[-1:]

    xy2_odd = xy2_odd[(np.abs(xy2_odd - xy1_odd) > (size_a[-1] + size_b[-1])/2).any(1)]
    xy2_odd = xy2_odd[0:1]

    s1 = Shape(gap_max=0.01, hole_radius=0.2)
    s2 = Shape(gap_max=0.01)
    shapes.append([s1,s2])
    xy2 = np.concatenate([np.array(xy2)*size_a[:-1,None] + xy1[:-1],xy2_odd], 0)

    xy = np.stack([xy1, xy2], axis=1)
    size = np.stack([size_a, size_b], axis=1)

    if 'c' in condition:
        color = sample_random_colors(n_samples * 2)
        color1 = [np.ones([1, 3]) * color[i:i+1] for i in range(n_samples)]
        color2 = [np.ones([1, 3]) * color[i:i+1] for i in range(n_samples, 2*n_samples)]
        color = np.concatenate([color1, color2], axis=1)
    else:
        color = sample_random_colors(1)
        color = [np.ones([2, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shapes, color


def sn_task_contact(condition='xysc'):

    n_samples = 4
    n_objects = 2

    # image and object parameters
    internal_frame = 0.8
    pad = (1-internal_frame)/2

    max_size = 0.4
    min_size = max_size/2

    size = np.random.rand(n_samples, 2) * (max_size - min_size) + min_size

    shape = []
    xy = []
    for i in range(n_samples):

        s1 = Shape()
        s1.randomize()
        s2 = Shape()
        s2.randomize()

        if i == n_samples-1:
            xy_ = np.random.rand(2,2) * (1-size[i,:,None]) + size[i,:,None]/2
            while not (np.abs(xy_[0] - xy_[1]) - size[i].sum()/2 > 0).any():
                xy_ = np.random.rand(2,2) * (1-size[i,:,None]) + size[i,:,None]/2

            xy.append(xy_)

        else:
            positions, clump_size = sample_contact_many([s1, s2], size[i])

            xy0 = np.random.rand(2) * (1-clump_size) + clump_size/2
            xy_ = positions + xy0[None,:]

            xy.append(xy_)

            # dir_ = np.random.randint(4)
            # xy2_r = sample_contact(s1, s2, size[i,1]/size[i,0], direction=dir_)

            # xy2_r = xy2_r * size[i,0]

            # bb = np.stack([- size[i,0]*np.ones(2)/2, size[i,0]*np.ones(2)/2, xy2_r - size[i,1]/2, xy2_r + size[i,1]/2], 0)
            # bb = np.stack([bb.min(0), bb.max(0)], 0)
            # xy1_r = - bb.mean(0)
            # xy2_r = xy2_r + xy1_r
            # bb = bb[1] - bb[0]

            # xy_ = np.random.rand(2) * (1-bb) + bb/2

            # xy1 = xy1_r + xy_
            # xy2 = xy2_r + xy_

            # xy.append(np.stack([xy1, xy2], 0))

        shape.append([s1,s2])

    xy = np.stack(xy, 0)


    if 'c' in condition:
        color = sample_random_colors(n_samples * 2)
        color1 = [np.ones([1, 3]) * color[i:i+1] for i in range(n_samples)]
        color2 = [np.ones([1, 3]) * color[i:i+1] for i in range(n_samples, 2*n_samples)]
        color = np.concatenate([color1, color2], axis=1)
    else:
        color = sample_random_colors(1)
        color = [np.ones([2, 3]) * color for i in range(n_samples)]

    return xy, size, shape, color