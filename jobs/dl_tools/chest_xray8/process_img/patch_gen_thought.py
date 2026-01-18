import numpy as np 

'''mini version. experiment with patch generator parallelization here. '''

def get_subimg_inds(img_size=1024, stride=16):
    pad_up = img_size // stride
    padded_shape = (img_size + pad_up, img_size + pad_up)

    inds = []
    for y in range(0, padded_shape[0] // stride, stride):
        for x in range(0, padded_shape[1] // stride, stride):
            inds.append((y, y+img_size, x, x+img_size))
    print(inds)
    return np.array(inds), pad_up

if __name__ == '__main__':
    processes = 2
    img_size = 1024
    inds, pad_up  = get_subimg_inds(img_size = img_size, stride = 22)
    
    print('number of large image reps, bbox ',inds.shape)
    # this one already distributes the chunks pretty well
    chunks = np.array_split(inds, processes)
    for c in chunks:
        print(c.shape)

    padimg0 = np.zeros((img_size + pad_up, img_size + pad_up))

    print('pad img shape ', padimg0.shape)
    
    
    