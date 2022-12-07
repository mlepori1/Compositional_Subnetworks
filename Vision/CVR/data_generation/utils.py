import os
import numpy as np

from PIL import Image

import cv2


def cat_lists(lists):
    o = []
    for l in lists:
        o += l
    return o


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0) # assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0: 
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

# helper functions
def sample_position_inside_1(s1, s2, scale):
    c1 = s1.get_contour()
    c2 = s2.get_contour()
    
    c2 = c2 * scale
    bb_2 = c2.max(0) - c2.min(0)

    # sampling points
    range_ = (c1.max(0) - c1.min(0) - bb_2)
    starting = (c1.min(0) + bb_2/2)
    samples = np.random.rand(100, 2) * range_[None,:] + starting[None,:]

    p1c = np.concatenate([c1[:-1], c1[1:]], 1)[None,:,:]
    samples = samples[:,None,:]
    res = np.logical_and(
        np.logical_or(
            p1c[:,:,0:1] < samples[:,:,0:1], 
            p1c[:,:,2:3] < samples[:,:,0:1]), 
        np.logical_xor(
            p1c[:,:,1:2] <= samples[:,:,1:2], 
            p1c[:,:,3:4] <= samples[:,:,1:2])
        )[:,:,0]
    res1 = (res.sum(1)%2==1)
    res2 = (np.abs(samples - c1) > bb_2[None,None,:]/2).any(2).all(1)

    res = np.logical_and(res1, res2)

    samples = samples[res,0]

    return samples

def sample_position_inside_many(s1, shapes, scales):
    # Contour is defined as a set of X, Y coordinates with continuous values (to be perfectly precise in contour)
    c1 = s1.get_contour() # contour of the big shape
    c2s = [s2.get_contour() for s2 in shapes] # Contour of set of objects ot fit inside
    
    bbs_2 = np.array([c2.max(0) - c2.min(0) for c2 in c2s]) * np.array(scales)[:,None] # Get bounding boxes of the shapes
    
    n_shapes = len(shapes)

    # sampling points
    # Sample points inside the object, rejection sampling
    # Reject combinations of points with overlapping bounding boxes or are outside the object. 
    # Need to reject points that are too close to border

    # Range that you accept for samplign points
    ranges_ = (c1.max(0)[None,:] - c1.min(0)[None,:] - bbs_2) # Bounding box for object 1 - bounding box from object 2, avoiding objects that overlap with border
    starting = (c1.min(0)[None,:] + bbs_2/2) # Pad box where you're samping
    samples = np.random.rand(500, n_shapes, 2) * ranges_[None, :, :] + starting[None, :, :] # Sample points, sample 500 points at first

    dists = np.abs(samples[:,:,None,:] - samples[:,None,:,:]) - (bbs_2[None,:,None,:] + bbs_2[None,None,:,:])/2 > 0 # Distances between the inside objects
    triu_idx = np.triu_indices(n_shapes, k=1)[0]*n_shapes + np.triu_indices(n_shapes, k=1)[1] # combinations of center points of internal objects that don't overlap
    no_overlap = dists.any(3).reshape(500, n_shapes*n_shapes)[:, triu_idx].all(1) # Sample N shape points and x and y
    
    samples = samples[no_overlap] # By now, you know your object's bounding box is within the box and that they have'nt touched each other

    n_samples_left = len(samples)
    bb_2_ = np.concatenate([bbs_2]*n_samples_left, 0) # Now, you need to know whether your inside object is actually inside the object, not just in the boudning box
    
    samples = samples.reshape([n_samples_left*n_shapes, 2]) # all objects x, y coordinates
    
    p1c = np.concatenate([c1[:-1], c1[1:]], 1)[None,:,:] # objects are defined by segments, there are 80 segments in the shape. P1C is the list of segments in the enclosing shape
    samples = samples[:,None,:]
    res = np.logical_and( # For each sampled inner object, does the larger shape enclose the inner shape? If you draw a line in one direction from your small shape and it 
                            # touches the larger shape 0 or 2 times, then you're outside the large shape, if you touch once, then you're inside the shape
        np.logical_or(
            p1c[:,:,0:1] < samples[:,:,0:1], 
            p1c[:,:,2:3] < samples[:,:,0:1]), 
        np.logical_xor(
            p1c[:,:,1:2] <= samples[:,:,1:2], 
            p1c[:,:,3:4] <= samples[:,:,1:2])
        )[:,:,0]
    res1 = (res.sum(1)%2==1) 
    res2 = (np.abs(samples - c1[None,:,:]) > bb_2_[:,None,:]/2).any(2).all(1) # This part verifies that the contours do not touch the bounding box of the larger shape
    
    res = np.logical_and(res1, res2) # Verifies that both are true, that you're inside the object and not touching the bounding box
    res = res.reshape([-1, n_shapes]).all(1) # All the shapes that pass, return them
    samples = samples.reshape([-1, n_shapes, 2])

    # samples = samples[res,0]
    samples = samples[res]
        
    return samples


def sample_int_sum_n(n_numbers, s, min_v=0):
    samples = np.random.rand(n_numbers)
    samples = samples/samples.sum()*s
    samples = np.ceil(samples).astype(int)
    samples[samples<min_v] = min_v
    
    while samples.sum()>s:
        diff = samples.sum() - s    
        idx = np.where(samples>min_v)[0]
        if diff<len(idx):
            idx = np.random.choice(idx, size=diff, replace=False)
        samples[idx] -=1
    return samples    


# different n values that cover a range without overlapping with minimum distances between them
def sample_over_range(range_, min_dists):
    n_values = len(min_dists)
    
    dists = np.random.rand(n_values)
    dists = dists / dists.sum() * (range_[1] - range_[0] - min_dists.sum())
    dists[0] = dists[0] * np.random.rand()
    dists = dists + min_dists
    v = np.cumsum(dists)
    v = v - min_dists[0]/2 + range_[0]

    return v

def sample_over_range_t(n_samples, range_, min_dists):
    if len(range_.shape) == 1:
        range_ = range_[None,:]
    if len(min_dists.shape) == 1:
        min_dists = min_dists[None,:]

    n_values = min_dists.shape[1]
    
    dists = np.random.rand(n_samples, n_values)
    dists = dists / dists.sum(1)[:,None] * (range_[:,1] - range_[:,0] - min_dists.sum(1)[:,None])
    dists[:,0] = dists[:,0] * np.random.rand(n_samples)
    dists = dists + min_dists
    v = np.cumsum(dists, 1)
    v = v - min_dists/2 + range_[:,0:1]

    return v


def sample_positions(size, n_sample_min=1, max_tries=10, n_samples_over=100):
    max_tries = 10
    i = 0
    n_samples_over = 100

    n_objects = size.shape[1]

    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]
    xy_ = []
    xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
    valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
    if valid.any():
        xy_ = xy[valid][:n_sample_min]

    while  len(xy_) < n_sample_min and i<max_tries:
        xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
        valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
        if valid.any():
            if len(xy_)==0:
                xy_ = xy[valid][:n_sample_min-len(xy_)]
            else:
                xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)

        i+=1

    if len(xy_) == 0:
        xy_ = xy[:n_sample_min]
    elif len(xy_) < n_sample_min:
        xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)
        
    return xy_



def sample_positions_bb(size, n_sample_min=1, max_tries=10, n_samples_over=100):
    max_tries = 10
    i = 0
    n_samples_over = 100

    n_objects = size.shape[1]

    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]
    xy_ = []
    xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size) + size/2
    valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,:]+size[:,None,:,:])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
    if valid.any():
        xy_ = xy[valid][:n_sample_min]

    while  len(xy_) < n_sample_min and i<max_tries:
        xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size) + size/2
        valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,:]+size[:,None,:,:])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
        if valid.any():
            if len(xy_)==0:
                xy_ = xy[valid][:n_sample_min-len(xy_)]
            else:
                xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)

        i+=1

    if len(xy_) == 0:
        xy_ = xy[:n_sample_min]
    elif len(xy_) < n_sample_min:
        xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)
        
    return xy_


def sample_random_colors(n_samples):
    h = np.random.rand(n_samples)
    s = np.random.rand(n_samples) * 0.5 + 0.5
    v = np.random.rand(n_samples) * 1

    color = np.stack([h,s,v],1)
    return color


def sample_shuffle_unshuffle_indices(n):
    perm = np.random.permutation(n)
    indices_input = np.arange(n)
    indices_output = indices_input[perm]
    rev_perm = (indices_output[:, None] == indices_input).argmax(axis=0)
    return perm, rev_perm


def shuffle_t(t, perms):
    # t.reshape()
    for i in range(t.shape[0]):
        t[i] = t[i, perms[i]]


# Shape has parameter (inner radius), which determines smallest inner circle that can fit inside the bigger 
# object. 

# Size is absolute, scale is relative to other shapes
def sample_contact(s1, s2, scale, direction=0):

    c1 = s1.get_contour()
    c2 = s2.get_contour()
    
    c2 = c2 * scale
    
    if direction==0:
        p1 = np.argmax(c1[:,0]) 
        p2 = np.argmin(c2[:,0]) 
    elif direction==1:
        p1 = np.argmin(c1[:,0]) 
        p2 = np.argmax(c2[:,0]) 
    elif direction==2:
        p1 = np.argmax(c1[:,1]) 
        p2 = np.argmin(c2[:,1]) 
    elif direction==3:
        p1 = np.argmin(c1[:,1]) 
        p2 = np.argmax(c2[:,1]) 

    xy2 = (c2.max(0) + c2.min(0))/2 - c2[p2] + c1[p1]
    # xy1 = np.zeros(2)
    
    return xy2

# NOTE Larger object is of size 1 for this sampling, gotta scale up the returned points by the actual size of the object
# NOTE Should be able to just replace the task insideness sample insidenesss with the new sample_contact_insideness function
def sample_contact_insideness(s1, s2, size2, a=0):
    c1 = s1.get_contour()
    c2 = s2.get_contour()
    
    c2 = c2 * size2
    bb_2 = c2.max(0) - c2.min(0)

    # sampling points
    range_ = (c1.max(0) - c1.min(0) - bb_2)
    starting = (c1.min(0) + bb_2/2)
    samples = np.random.rand(100, 2) * range_[None,:] + starting[None,:]

    p1c = np.concatenate([c1[:-1], c1[1:]], 1)[None,:,:]
    samples = samples[:,None,:]
    res = np.logical_and(
        np.logical_or(
            p1c[:,:,0:1] < samples[:,:,0:1], 
            p1c[:,:,2:3] < samples[:,:,0:1]), 
        np.logical_xor(
            p1c[:,:,1:2] <= samples[:,:,1:2], 
            p1c[:,:,3:4] <= samples[:,:,1:2])
        )[:,:,0]
    res1 = (res.sum(1)%2==1)
    res2 = (np.abs(samples - c1) > bb_2[None,None,:]/2).any(2).all(1)

    res = np.logical_and(res1, res2)

    samples = samples[res,0]

    
    # c1 = s1.get_contour()
    # c2 = s2.get_contour()
    
    ############## 1 sample version
    # # sample direction
    if isinstance(a, float):
        angle = a
    else:
        angle = np.random.rand(1) * 2 * np.pi

    if len(samples) == 0:
        return None

    pos2 = samples[0]

    # idx_p_contact_c1 = (c1 * np.array((np.cos(angle),np.sin(angle))).reshape(-1)).sum(-1) > 0
    # idx_p_contact_c2 = (c2 * np.array((np.cos(angle),np.sin(angle))).reshape(-1)).sum(-1) > 0 # Flipped from when you make something in contact and outside
    # idx_p_contact_c1 = np.ones(idx_p_contact_c1.shape, dtype=bool) # Temporarily get rid of this
    # idx_p_contact_c2 = np.ones(idx_p_contact_c2.shape, dtype=bool)
    
    # move object in direction
    c = c2 + pos2
    
    idx_min = np.linalg.norm(c1[:,None,:] - c[None,:,:], axis=2).argmin()
    s_ = len(c2) 
    idx_min_c1, idx_min_c2 = idx_min // s_, idx_min % s_
    p_clump = c1[idx_min_c1]
    p_obj = c2[idx_min_c2]
    new_pos = (p_clump - p_obj)*(1-4/128)
    # ############## 
    

    ############## N sample version
    # sample direction
    # if isinstance(a, float):
    #     angle = a
    # else:
    #     angle = np.random.rand(samples.shape[0]) * 2 * np.pi
    
    # angle = angle[:,None, None]

    # pos2 = samples

    # idx_p_contact_c1 = (c1[None,:, :] * (np.cos(angle),np.sin(angle))).sum(-1) > 0
    # idx_p_contact_c2 = (c2[None,:, :] * (np.cos(angle),np.sin(angle))).sum(-1) > 0
    
    # new_positions=[]
    # # move object in direction
    # c = c2[None,:,:] + pos2[:,None,:]
    # for i in range(samples.shape[0]):
    #     idx_min = np.linalg.norm(c1[idx_p_contact_c1[i]][:,None,:] - c[idx_p_contact_c2[i]][None,:,:], axis=2).argmin()
    #     s_ = idx_p_contact_c2.sum()
    #     idx_min_c1, idx_min_c2 = idx_min // s_, idx_min % s_
    #     p_clump = c1[idx_p_contact_c1][idx_min_c1]
    #     p_obj = c2[idx_p_contact_c2][idx_min_c2]
    #     new_pos = (p_clump - p_obj)*(1-4/128)
    #     new_positions.append(new_pos)
    # new_positions = np.stack(new_positions)

    return new_pos

# Intuition, sample an angle between two objects.
# Displace one object in the image so that the relative angle between the two is alpha
# Find the min point between the two shapes in this configuration
# Move the one object towards the other such that they touch
def sample_contact_many(shapes, sizes, a=None):
    n_objects = len(shapes)
    contours = [shapes[i].get_contour() * sizes[i] for i in range(n_objects)]

    # intialize clump as the first object
    clump = contours[0]
    positions = np.zeros([1,2])
    clump_size = np.ones(2) * sizes[0]
    for i in range(1, n_objects): # For all of the other objects
        # sample direction
        if a is None:
            angle = np.random.rand() * 2 * np.pi
        # Should always be this?
        elif isinstance(a, float):
            angle = a
        # Unless you input an a
        else:
            angle = a[i]

        # Pos2 is (x,y) coordinates of the second object, calculated using trig on the angle and size of new shape + size of clump  
        # You're displacing the object away from the clump, and in the right direction
        pos2 = (sizes[i]+clump_size) * np.array([np.cos(angle), np.sin(angle)])[None,:]
        
        # Only sample points on the side of the clump/object that are in the right direction. This is an optimizaiton
        # so you don't sample points in your distance calculation that couldn't possibly be the closest in a given direction.
        # Only points on the correct side of an orthogonal (to the angle of interest) bisecting line for an object.
        idx_p_contact_clump = (clump * (np.cos(angle),np.sin(angle))).sum(-1) > 0
        idx_p_contact_object = (contours[i] * (np.cos(angle),np.sin(angle))).sum(-1) < 0 # Note, this is flipped when the smaller object is inside the larger
        
        # move object in direction
        c = contours[i] + pos2
        
        # Find minimum distance between contours of object
        idx_min = np.linalg.norm(clump[idx_p_contact_clump][:,None,:] - c[idx_p_contact_object][None,:,:], axis=2).argmin() # Find pairwise closest points between two objects
        s_ = idx_p_contact_object.sum() # 
        idx_min_clump, idx_min_object = idx_min // s_, idx_min % s_
        p_clump = clump[idx_p_contact_clump][idx_min_clump]
        p_obj = contours[i][idx_p_contact_object][idx_min_object]
        new_pos = (p_clump - p_obj)*(1-4/128) # Move object to clump by the distance vector between the closest  between obj and clump, making exactly one point of contact between the two
        
        clump = np.concatenate([clump, contours[i]+new_pos[None,:]], 0) # Clump now incorporates the new object
        bb = clump.min(0), clump.max(0) # Calc the new bounding box of the clump
        
        clump = clump - (bb[1] + bb[0])/2
        clump_size = bb[1] - bb[0] # Size of the bounding box of the clump

        positions = np.concatenate([positions,new_pos[None,:]], 0) # absolute positions of the shapes that have been clumped
        positions = positions - (bb[1] + bb[0])/2 # Ensures that this is a relative position, irrespective of bounding box padding, will sample center of the clump in larger function

    return positions, clump_size

def flip_diag_scene(xys, shapes):

    for s in shapes:
        s.flip_diag()

    for i, xy in enumerate(xys):
        xys[i] = xy[::-1]

    return xys, shapes

def render_cv(xy, size, shapes, color=None, image_size=128):
        
    color = [hsv_to_rgb(c[0], c[1], c[2]) for c in color]

    image = (np.ones([image_size,image_size, 3]) * 255).astype(np.uint8)

    for i in range(len(shapes)):
        size_ = size[i]
        s_ = shapes[i]
        s_.scale(size_)
        xy_ = xy[i]

        c = s_.get_contour()
        
        c = (c*image_size).astype(int)

        c_ = np.concatenate([c,c[0:1]],0)
        dist = np.abs(c_[1:] - c_[:-1]) 
        c = c[(dist>0).any(1)]

        c = c + (xy_[None,:] * image_size).astype(int)

        col_ = (np.array(color[i])*255).tolist()
        cv2.drawContours(image, [c], -1, col_, 1)
        
    return image


def render_ooo(xy, size, shape, color, image_size=128):

    images = []
    for i in range(len(shape)):
        im = render_cv(xy[i], size[i], shape[i], color[i], image_size=128)
        im = np.pad(im, [[4,4], [4,4], [0,0]], constant_values=0)
        images.append(im)

    images = np.concatenate(images, axis=1)
    
    return images


def save_image_human_exp(images, meta, base_path):
    im_shape = images.shape
    
    dim_0 = im_shape[1]//4

    pad = (dim_0-128)//2

    images = images.reshape([im_shape[0]//dim_0, dim_0, 4, dim_0, 3]).transpose([0,2,1,3,4]).reshape([-1, dim_0, dim_0, 3])
    for i in range(len(images)):
        idx1, idx2 = i//4, i%4 
        save_path = os.path.join(base_path, '{:02d}_{}.png'.format(idx1, idx2))
        img = Image.fromarray(images[i,pad:dim_0-pad, pad:dim_0-pad]).convert('RGB')
        img.save(save_path)


def save_image_bin(images, base_path, task_name):
    save_path = base_path + '{}.png'.format(task_name)
    if images.dtype != np.uint8:
        images = (images*255).astype(np.uint8)
    img = Image.fromarray(images).convert('1')
    img.save(save_path)



def save_image(images, base_path, task_name):
    save_path = base_path + '{}.png'.format(task_name)
    # if images.dtype != np.uint8:
    #     images = (images*255).astype(np.uint8)
    img = Image.fromarray(images).convert('RGB')
    img.save(save_path)

