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

def sn_task_1_contact_inside(condition='c', odd_one_out = "all"):
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



def sn_task_1_contact(condition='c'):
    """
    The rule is that 3 images contain 2 objects, one of which is in contact with the other

    There are three types of images that can occur:
        (inside_contact) Inner object is inside and in contact with enclosing object (+)
        (no_inside_contact) Two objects are in contact, but one is not enclosing the other (+)
        (inside_no_contact) Inner object is inside but not in contact with enclosing object (-)
        (no_inside_no_contact) Two objects are neither in contact nor enclosing one another (-)
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

        rule_instance = np.random.choice(["inside_contact", "no_inside_contact"])

        if rule_instance == "inside_contact":

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

            xy2.append(sample * size_a[i] + xy1[i])
            shapes.append([s1, s2])
        
        elif rule_instance == "no_inside_contact":
            # This is just the normal contact function
            s1 = Shape(gap_max=0.07, hole_radius=0.2)
            s2 = Shape(gap_max=0.01)
            shape_sizes = [size_a[i], size_b[i]]
            positions, clump_size = sample_contact_many([s1, s2], shape_sizes) # Positions get you the offsets for both object 1 and object 2

            xy1_odd_init = np.random.rand(2) * (1-clump_size) + clump_size/2 # Need to resample the initial position of object 1 now that you                                                              
                                                                        # know the clump size
            
            # Apply offsets
            xy2_odd = positions[-1, :] + xy1_odd_init 
            xy1_odd = positions[0, :] + xy1_odd_init

            xy1[i] = xy1_odd
            shapes.append([s1, s2])
            xy2.append(xy2_odd)
    
    # Define which way we will break the rule to create the odd one out
    odd_one_out = np.random.choice(["inside_no_contact", "no_inside_no_contact"])

    # Sample new shapes of the same form
    s1 = Shape(gap_max=0.07, hole_radius=0.2)
    s2 = Shape(gap_max=0.01)

    if odd_one_out == "no_inside_no_contact":
        # The opposite of the normal inside function is no_inside_no_contact
        range_2 = 1 - size_b[-1]
        starting_2 = size_b[-1]/2
        xy2_odd = np.random.rand(100,2) * range_2 + starting_2
        xy1_odd = xy1[-1:]

        xy2_odd = xy2_odd[(np.abs(xy2_odd - xy1_odd) > (size_a[-1] + size_b[-1])/2).any(1)]
        xy2_odd = xy2_odd[0:1]

        shapes.append([s1,s2])
        xy2 = np.concatenate([np.array(xy2),xy2_odd], 0)

    elif odd_one_out == "inside_no_contact":
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
        xy2 = np.concatenate([np.array(xy2), xy2_odd.reshape(1, 2)], 0)
        shapes.append([s1, s2])

 
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



def sn_task_1_contact_adversarial_inside(condition='c'):
    """
    The rule is that 3 images contain 2 objects, one of which is in contact with the other
    This function generates images that are adversarial to insideness subnetworks
    There are three types of images that can occur:
        (inside_contact) Inner object is inside and in contact with enclosing object (+)
        (no_inside_contact) Two objects are in contact, but one is not enclosing the other (+)
        (inside_no_contact) Inner object is inside but not in contact with enclosing object (-)
        (no_inside_no_contact) Two objects are neither in contact nor enclosing one another (-)
    
    In a given set, all images will have the same insideness properties
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

    insideness = np.random.choice(["inside", "no_inside"])

    for i in range(n_samples-1):

        if insideness == "inside":
            # This is inside_contact
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

            xy2.append(sample * size_a[i] + xy1[i])
            shapes.append([s1, s2])
        
        elif insideness == "no_inside":
            # This is no_inside_contact
            # This is just the normal contact function
            s1 = Shape(gap_max=0.07, hole_radius=0.2)
            s2 = Shape(gap_max=0.01)
            shape_sizes = [size_a[i], size_b[i]]
            positions, clump_size = sample_contact_many([s1, s2], shape_sizes) # Positions get you the offsets for both object 1 and object 2

            xy1_odd_init = np.random.rand(2) * (1-clump_size) + clump_size/2 # Need to resample the initial position of object 1 now that you                                                              
                                                                        # know the clump size
            
            # Apply offsets
            xy2_odd = positions[-1, :] + xy1_odd_init 
            xy1_odd = positions[0, :] + xy1_odd_init

            xy1[i] = xy1_odd
            shapes.append([s1, s2])
            xy2.append(xy2_odd)
    

    # Sample new shapes of the same form
    s1 = Shape(gap_max=0.07, hole_radius=0.2)
    s2 = Shape(gap_max=0.01)

    if insideness == "no_inside":
        # This is no_inside_no_contact
        # The opposite of the normal inside function is no_inside_no_contact
        range_2 = 1 - size_b[-1]
        starting_2 = size_b[-1]/2
        xy2_odd = np.random.rand(100,2) * range_2 + starting_2
        xy1_odd = xy1[-1:]

        xy2_odd = xy2_odd[(np.abs(xy2_odd - xy1_odd) > (size_a[-1] + size_b[-1])/2).any(1)]
        xy2_odd = xy2_odd[0:1]

        shapes.append([s1,s2])
        xy2 = np.concatenate([np.array(xy2),xy2_odd], 0)

    elif insideness == "inside":
        # This is inside_no_contact
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
        xy2 = np.concatenate([np.array(xy2), xy2_odd.reshape(1, 2)], 0)
        shapes.append([s1, s2])

 
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


def sn_task_1_inside(condition='c'):
    """
    The rule is that 3 images contain 2 objects, one of which is inside the other

    There are three types of images that can occur:
        (inside_contact) Inner object is inside and in contact with enclosing object (+)
        (no_inside_contact) Two objects are in contact, but one is not enclosing the other (-)
        (inside_no_contact) Inner object is inside but not in contact with enclosing object (+)
        (no_inside_no_contact) Two objects are neither in contact nor enclosing one another (-)
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

        rule_instance = np.random.choice(["inside_contact", "inside_no_contact"])

        if rule_instance == "inside_contact":

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

            xy2.append(np.array(sample * size_a[i] + xy1[i]).reshape(-1))
            shapes.append([s1, s2])

        elif rule_instance == "inside_no_contact":
            # this is just the normal insideness function
            done = False

            for _ in range(max_attempts):
                # In sample position inside many, the scale array is used to determine bounding boxes
                # around objects where objects shouldnt be sampled from. I artificially increase this
                # to prevent any inner objects from touching the outer object.
                # The function already does this, but it operates in a continuous x,y coordinate space,
                # which may result in objects touching once things are aliased in image pixels.
                # This hack ensures that you never run into this edge case.
                s1 = Shape(gap_max=0.07, hole_radius=0.2)
                s2 = Shape(gap_max=0.01)
                samples = sample_position_inside_many(s1, [s2], [(size_b[i]/size_a[i]) + .2])
                if len(samples)>0:
                    done = True

                if done:
                    break
                else:
                    s1.randomize()
                    s2.randomize()

            if not done:
                return np.zeros([100,100])

            xy2_new = samples[0]
            xy2_new = np.array(xy2_new)*size_a[i] + xy1[i]
            xy2.append(xy2_new.reshape(-1))
            shapes.append([s1, s2])
    
    # Define which way we will break the rule to create the odd one out
    odd_one_out = np.random.choice(["no_inside_contact", "no_inside_no_contact"])

    # Sample new shapes of the same form
    s1 = Shape(gap_max=0.07, hole_radius=0.2)
    s2 = Shape(gap_max=0.01)

    if odd_one_out == "no_inside_no_contact":
        # The opposite of the normal inside function is no_inside_no_contact
        range_2 = 1 - size_b[-1]
        starting_2 = size_b[-1]/2
        xy2_odd = np.random.rand(100,2) * range_2 + starting_2
        xy1_odd = xy1[-1:]

        xy2_odd = xy2_odd[(np.abs(xy2_odd - xy1_odd) > (size_a[-1] + size_b[-1])/2).any(1)]
        xy2_odd = xy2_odd[0:1]

        shapes.append([s1,s2])
        xy2.append(xy2_odd.reshape(-1))

    elif odd_one_out == "no_inside_contact":
        # This is just the normal contact function
        s1 = Shape(gap_max=0.07, hole_radius=0.2)
        s2 = Shape(gap_max=0.01)
        shape_sizes = [size_a[-1], size_b[-1]]
        positions, clump_size = sample_contact_many([s1, s2], shape_sizes) # Positions get you the offsets for both object 1 and object 2

        xy1_odd_init = np.random.rand(2) * (1-clump_size) + clump_size/2 # Need to resample the initial position of object 1 now that you                                                              
                                                                    # know the clump size
        
        # Apply offsets
        xy2_odd = positions[-1, :] + xy1_odd_init 
        xy1_odd = positions[0, :] + xy1_odd_init

        xy1[-1] = xy1_odd

        shapes.append([s1, s2])
        xy2.append(xy2_odd.reshape(-1))


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


def sn_task_1_inside_adversarial_contact(condition='c'):
    """
    The rule is that 3 images contain 2 objects, one of which is inside the other

    There are three types of images that can occur:
        (inside_contact) Inner object is inside and in contact with enclosing object (+)
        (no_inside_contact) Two objects are in contact, but one is not enclosing the other (-)
        (inside_no_contact) Inner object is inside but not in contact with enclosing object (+)
        (no_inside_no_contact) Two objects are neither in contact nor enclosing one another (-)

    This dataset is adversarial to contact subnetworks. all images in a given set have same contact properties
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

    contact = np.random.choice(["contact", "no_contact"])

    for i in range(n_samples-1):

        if contact == "contact":
            # This is inside contact
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

            xy2.append(np.array(sample * size_a[i] + xy1[i]).reshape(-1))
            shapes.append([s1, s2])

        elif contact == "no_contact":
            # This is inside_no_contact
            # this is just the normal insideness function
            done = False

            for _ in range(max_attempts):
                # In sample position inside many, the scale array is used to determine bounding boxes
                # around objects where objects shouldnt be sampled from. I artificially increase this
                # to prevent any inner objects from touching the outer object.
                # The function already does this, but it operates in a continuous x,y coordinate space,
                # which may result in objects touching once things are aliased in image pixels.
                # This hack ensures that you never run into this edge case.
                s1 = Shape(gap_max=0.07, hole_radius=0.2)
                s2 = Shape(gap_max=0.01)
                samples = sample_position_inside_many(s1, [s2], [(size_b[i]/size_a[i]) + .2])
                if len(samples)>0:
                    done = True

                if done:
                    break
                else:
                    s1.randomize()
                    s2.randomize()

            if not done:
                return np.zeros([100,100])

            xy2_new = samples[0]
            xy2_new = np.array(xy2_new)*size_a[i] + xy1[i]
            xy2.append(xy2_new.reshape(-1))
            shapes.append([s1, s2])
    
    # Sample new shapes of the same form
    s1 = Shape(gap_max=0.07, hole_radius=0.2)
    s2 = Shape(gap_max=0.01)

    if contact == "no_contact":
        # This is no_inside_no_contact
        # The opposite of the normal inside function is no_inside_no_contact
        range_2 = 1 - size_b[-1]
        starting_2 = size_b[-1]/2
        xy2_odd = np.random.rand(100,2) * range_2 + starting_2
        xy1_odd = xy1[-1:]

        xy2_odd = xy2_odd[(np.abs(xy2_odd - xy1_odd) > (size_a[-1] + size_b[-1])/2).any(1)]
        xy2_odd = xy2_odd[0:1]

        shapes.append([s1,s2])
        xy2.append(xy2_odd.reshape(-1))

    elif contact == "contact":
        # This is no_inside_contact
        # This is just the normal contact function
        s1 = Shape(gap_max=0.07, hole_radius=0.2)
        s2 = Shape(gap_max=0.01)
        shape_sizes = [size_a[-1], size_b[-1]]
        positions, clump_size = sample_contact_many([s1, s2], shape_sizes) # Positions get you the offsets for both object 1 and object 2

        xy1_odd_init = np.random.rand(2) * (1-clump_size) + clump_size/2 # Need to resample the initial position of object 1 now that you                                                              
                                                                    # know the clump size
        
        # Apply offsets
        xy2_odd = positions[-1, :] + xy1_odd_init 
        xy1_odd = positions[0, :] + xy1_odd_init

        xy1[-1] = xy1_odd

        shapes.append([s1, s2])
        xy2.append(xy2_odd.reshape(-1))


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


def sn_task_2_inside_count(condition='c', odd_one_out = "no_inside"):
    """
    The rule is that 3 images contain N big objects and N small objects. Each big object contains a small object.

    odd_one_out determines which type of rule breaking is occuring in the odd one out.
    There are three types of breaking that can occur:
        (no_count) N +/- 1 Nested objects
        (no_inside) N-1 Nested Objects, 1 big and 1 small object not nested
        (no_count_inside) N +/- 2 Nested objects, 1 big and 1 small object not nested
    Option "all" randomly samples these instances for generating the odd one out
    """
    max_attempts = 20

    n_samples = 4
    n_samples_over = 100

    n_object_pairs = np.random.randint(low=2, high=4)
    n_rule_objects = 2 * 3 * n_object_pairs
    # n_objects = n_objects_samples.max()
    # min_size_obj = 0.2
    min_btw_size = 0.3


    all_xy = []
    all_size = []
    all_shape = []

    for i in range(n_samples - 1):
        # xy_i = xy[i,:n_objects_samples[i]]
        xy_ = []
        size_ = []
        shape_ = []

        done = False
        n_objects = n_object_pairs + 2 # To ensure that all shapes are drawn from the same 
            # distribution (even if there are two more shapes in the OOO condition), sample shape
            # sizes and positions as if there were N+2 objects in the image.

        min_size_obj = min(0.9/n_objects, 0.3)
        min_btw_size = min_size_obj

        triu_idx = np.triu_indices(n_objects, k=1)
        triu_idx = triu_idx[0]*n_objects + triu_idx[1]

        for _ in range(max_attempts):
            xy = np.random.rand(n_samples_over, n_objects, 2) * (1-min_size_obj) + min_size_obj/2
            no_overlap = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - min_btw_size>0).any(3).reshape([-1, n_objects*n_objects])[:,triu_idx].all(1)
            if no_overlap.sum()>0:
                done = True
                break

        if not done:
            print('')

        xy = xy[no_overlap][0]

        if n_objects >1: # Always true in this case
            non_diag = np.where(~np.eye(n_objects,dtype=bool))
            non_diag = non_diag[0]*n_objects + non_diag[1]
            dists_obj = np.abs(xy[:,None,:] - xy[None,:,:]).max(2).reshape([n_objects**2])[non_diag].reshape([n_objects, n_objects-1]).min(1)
            dists_edge = np.stack([xy, 1-xy],2).min(1).min(1)*2
            max_size = np.stack([dists_edge, dists_obj], 1).min(1)
        else:
            max_size = 0.6
        # max_size = np.stack([xy, 1-xy],2).min(2).min(2)
        min_size = max_size/2 # Consider constraining size range

        size = np.random.rand(n_objects) * (max_size - min_size) + min_size
        size_in = np.random.rand(n_objects) * (size/2.5 - size/4) + size/4

        # Now generate n_object_pairs shapes 
        for j in range(n_object_pairs):
            done = False
            s1 = Shape(gap_max=0.08, hole_radius=0.2)
            s2 = Shape(gap_max=0.01)
            for _ in range(max_attempts):

                samples = sample_position_inside_1(s1, s2, size_in[j]/size[j])
                if len(samples)>0:
                    done = True

                if done:
                    break
                else:
                    s1.randomize()
                    s2.randomize()

            if done:
                xy_in = samples[0]

                xy_.append(xy[j])
                shape_.append(s1)
                size_.append(size[j])

                xy_.append(xy_in * size[j] + xy[j])
                shape_.append(s2)
                size_.append(size_in[j])

        all_xy.append(xy_)
        all_size.append(size_)
        all_shape.append(shape_)

    # Define which way we will break the rule to create the odd one out
    if odd_one_out == "all":
        odd_one_out = np.random.choice(["no_count", "no_inside", "no_count_inside"])

    if odd_one_out == "no_count":
        # There is either one more or one less nested shapes

        n_odd_object_pairs = n_object_pairs + np.random.choice([-1, 1])
        n_odd_objects = n_odd_object_pairs * 2

        xy_ = []
        size_ = []
        shape_ = []

        done = False
        n_objects = n_object_pairs + 2 # To ensure that all shapes are drawn from the same 
            # distribution (even if there is another/one fewer shape in the OOO condition), sample shape
            # sizes and positions as if there were N+1 objects in the image.

        min_size_obj = min(0.9/n_objects, 0.3)
        min_btw_size = min_size_obj

        triu_idx = np.triu_indices(n_objects, k=1)
        triu_idx = triu_idx[0]*n_objects + triu_idx[1]

        for _ in range(max_attempts):
            xy = np.random.rand(n_samples_over, n_objects, 2) * (1-min_size_obj) + min_size_obj/2
            no_overlap = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - min_btw_size>0).any(3).reshape([-1, n_objects*n_objects])[:,triu_idx].all(1)
            if no_overlap.sum()>0:
                done = True
                break

        if not done:
            print('')

        xy = xy[no_overlap][0]

        if n_objects >1: # Always true in this case
            non_diag = np.where(~np.eye(n_objects,dtype=bool))
            non_diag = non_diag[0]*n_objects + non_diag[1]
            dists_obj = np.abs(xy[:,None,:] - xy[None,:,:]).max(2).reshape([n_objects**2])[non_diag].reshape([n_objects, n_objects-1]).min(1)
            dists_edge = np.stack([xy, 1-xy],2).min(1).min(1)*2
            max_size = np.stack([dists_edge, dists_obj], 1).min(1)
        else:
            max_size = 0.6
        # max_size = np.stack([xy, 1-xy],2).min(2).min(2)
        min_size = max_size/2 # Consider constraining size range

        size = np.random.rand(n_objects) * (max_size - min_size) + min_size
        size_in = np.random.rand(n_objects) * (size/2.5 - size/4) + size/4

        # Now generate n_object_pairs shapes 
        for j in range(n_odd_object_pairs):
            done = False
            s1 = Shape(gap_max=0.08, hole_radius=0.2)
            s2 = Shape(gap_max=0.01)
            for _ in range(max_attempts):

                samples = sample_position_inside_1(s1, s2, size_in[j]/size[j])
                if len(samples)>0:
                    done = True

                if done:
                    break
                else:
                    s1.randomize()
                    s2.randomize()

            if done:
                xy_in = samples[0]

                xy_.append(xy[j])
                shape_.append(s1)
                size_.append(size[j])

                xy_.append(xy_in * size[j] + xy[j])
                shape_.append(s2)
                size_.append(size_in[j])

        all_xy.append(xy_)
        all_size.append(size_)
        all_shape.append(shape_)

    if odd_one_out == "no_inside":
        # Same total number of shapes, but one pair is not nested

        xy_ = []
        size_ = []
        shape_ = []

        done = False
        n_objects = n_object_pairs + 2 # To ensure that all shapes are drawn from the same 
            # distribution (even if there are two more shapes in the OOO condition), sample shape
            # sizes and positions as if there were N+2 objects in the image. In this OOO condition
            # they are used to place the extra non-nested object!
        n_odd_objects = n_object_pairs * 2

        min_size_obj = min(0.9/n_objects, 0.3)
        min_btw_size = min_size_obj

        triu_idx = np.triu_indices(n_objects, k=1)
        triu_idx = triu_idx[0]*n_objects + triu_idx[1]

        for _ in range(max_attempts):
            xy = np.random.rand(n_samples_over, n_objects, 2) * (1-min_size_obj) + min_size_obj/2
            no_overlap = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - min_btw_size>0).any(3).reshape([-1, n_objects*n_objects])[:,triu_idx].all(1)
            if no_overlap.sum()>0:
                done = True
                break

        if not done:
            print('')

        xy = xy[no_overlap][0]

        if n_objects >1: # Always true in this case
            non_diag = np.where(~np.eye(n_objects,dtype=bool))
            non_diag = non_diag[0]*n_objects + non_diag[1]
            dists_obj = np.abs(xy[:,None,:] - xy[None,:,:]).max(2).reshape([n_objects**2])[non_diag].reshape([n_objects, n_objects-1]).min(1)
            dists_edge = np.stack([xy, 1-xy],2).min(1).min(1)*2
            max_size = np.stack([dists_edge, dists_obj], 1).min(1)
        else:
            max_size = 0.6
        # max_size = np.stack([xy, 1-xy],2).min(2).min(2)
        min_size = max_size/2 # Consider constraining size range

        size = np.random.rand(n_objects) * (max_size - min_size) + min_size
        size_in = np.random.rand(n_objects) * (size/2.5 - size/4) + size/4

        # Now generate n_object_pairs - 1 nested shapes
        for j in range(n_object_pairs - 1):
            done = False
            s1 = Shape(gap_max=0.08, hole_radius=0.2)
            s2 = Shape(gap_max=0.01)
            for _ in range(max_attempts):

                samples = sample_position_inside_1(s1, s2, size_in[j]/size[j])
                if len(samples)>0:
                    done = True

                if done:
                    break
                else:
                    s1.randomize()
                    s2.randomize()

            if done:
                xy_in = samples[0]

                xy_.append(xy[j])
                shape_.append(s1)
                size_.append(size[j])

                xy_.append(xy_in * size[j] + xy[j])
                shape_.append(s2)
                size_.append(size_in[j])

        # Now generate two non-nested objects
        done = False
        s1 = Shape(gap_max=0.08, hole_radius=0.2)
        s2 = Shape(gap_max=0.01)

        # We have already generated these xy values and sizes 
        # for 2 extra shapes, now use them
        s1_idx = n_object_pairs - 1
        s2_idx = n_object_pairs
        xy_.append(xy[s1_idx])
        shape_.append(s1)
        size_.append(size[s1_idx])

        xy_.append(xy[s2_idx]) # Place small object somewhere new
        shape_.append(s2)
        size_.append(size_in[s1_idx]) # hasn't been used yet


        all_xy.append(xy_)
        all_size.append(size_)
        all_shape.append(shape_)

    if 'c' in condition:
        color = sample_random_colors(n_rule_objects + n_odd_objects)

        shape_colors = np.array([np.ones([3]) * c for c in color])
        rule_objects = int(n_rule_objects / 3)
        print(rule_objects)
        print([len(s) for s in [shape_colors[i*rule_objects:i+1*rule_objects] for i in range(3)]])
        color1 = [np.stack(shape_colors[i*rule_objects:i+1*rule_objects], axis=0) for i in range(3)]
        color1.append(np.stack(shape_colors[3*rule_objects:], axis=0))
        print(len(color1))
        print(color1[0].shape)
        print("XX")
        color = [np.ones([n_objects + n_odd_objects, 3]) * color[i:i+1] for i in range(n_samples)]
        #print(color)
        print(len(color))
        print(color[0].shape)
        print("FDSA")
        '''        color = sample_random_colors(n_objects + n_odd_objects)
        print(color.shape)
        print(np.ones([n_objects + n_odd_objects, 3]).shape)
        color = np.ones([n_objects + n_odd_objects, 3]) * color
        print(color)
        '''

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_rule_objects + n_odd_objects, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    xy, size, shape = all_xy, all_size, all_shape

    return xy, size, shape, color