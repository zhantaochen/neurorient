import numpy as np

class RandomPatch:
    """ Randomly place num_patch patch with the size of size_y * size_x onto an image.
    """

    def __init__(self, num_patch, size_patch_y,    size_patch_x, 
                                  var_patch_y = 0, var_patch_x = 0, 
                                  returns_mask = False):
        self.num_patch    = num_patch                   # ...Number of patches
        self.size_patch_y = size_patch_y                # ...Size of the patch in y dimension
        self.size_patch_x = size_patch_x                # ...Size of the patch in x dimension
        self.var_patch_y  = max(0, min(var_patch_y, 1)) # ...Percent variation with respect to the patch size in x dimension
        self.var_patch_x  = max(0, min(var_patch_x, 1)) # ...Percent variation with respect to the patch size in y dimension
        self.returns_mask = returns_mask                # ...Is it allowed to return a mask

        return None


    def __call__(self, img):
        # Get the size of the image...
        size_img_y, size_img_x = img.shape[-2:]

        # Construct a mask of ones with the same size of the image...
        mask = np.ones_like(img)

        # Generate a number of random position...
        pos_y = np.random.randint(low = 0, high = size_img_y, size = self.num_patch)
        pos_x = np.random.randint(low = 0, high = size_img_x, size = self.num_patch)

        # Stack two column vectors to form an array of (x, y) indices...
        pos_y = pos_y.reshape(-1,1)
        pos_x = pos_x.reshape(-1,1)
        pos   = np.hstack((pos_y, pos_x))

        # Place patch of zeros at all pos as top-left corner...
        for (y, x) in pos:
            size_patch_y = self.size_patch_y
            size_patch_x = self.size_patch_x

            # Apply random variance...
            # Find the absolute max pixel to vary
            varsize_patch_y = int(size_patch_y * self.var_patch_y)
            varsize_patch_x = int(size_patch_x * self.var_patch_x)

            # Sample an integer from the min-max pixel to vary
            # Also allow user to set var_patch = 0 when variance is not desired
            delta_patch_y = np.random.randint(low = -varsize_patch_y, high = varsize_patch_y if varsize_patch_y else 1)
            delta_patch_x = np.random.randint(low = -varsize_patch_x, high = varsize_patch_x if varsize_patch_x else 1)

            # Apply the change
            size_patch_y += delta_patch_y
            size_patch_x += delta_patch_x

            # Find the limit of the bottom/right-end of the patch...
            y_end = min(y + size_patch_y, size_img_y)
            x_end = min(x + size_patch_x, size_img_x)

            # Patch the area with zeros...
            mask[:, y : y_end, x : x_end] = 0

        # Apply the mask...
        img_masked = mask * img

        # Construct the return value...
        # Parentheses are necessary
        output = img_masked if not self.returns_mask else (img_masked, mask)

        return output
