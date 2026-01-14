import numpy as np 

'''mini version. experiment with patch generator parallelization here. '''

def patch_generator(img, patch_size=128, stride=None):
    # do it this way so you can sample the patches instead of 
    # saving a list of patches to work on. 
    if stride is None:
        stride = patch_size

    H, W = img.shape[:2] 
    
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            yield img[i:i+patch_size, j:j+patch_size], i, j

def get_subimg_inds(img_size = 1024, stride = 8):
    pad_up = img_size//stride # keep in case you need to change the logic
    padimg0 = np.zeros((img_size + pad_up, img_size + pad_up))
    print('pad img shape ', padimg0.shape)
    inds = []
    for dy in range(0, padimg0.shape[0]//stride, stride):
        for dx in range(0, padimg0.shape[1]//stride, stride):
            inds.append([dy, dy+img_size, dx, dx+img_size])
            # print(padimg[dy:dy+img_size, dx:dx+img_size].shape) # yeah ok this part works. 
            # print('dx stuff : ', dx, dx+img_size)
            # print('dy stuff : ', dy, dy+img_size)
    
    # from here, you take the subimg_inds list. and split it up.
    return np.array(inds), padimg0

if __name__ == '__main__':
     

    processes = 4
    
    inds, padimg0 = get_subimg_inds(img_size = 1024, stride = 16)
    print('number of large image reps, bbox ',inds.shape)
    chunks = np.array_split(inds, processes)
    for c in chunks:
        print(c.shape)
    