import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob

output_images_path = './output_images/'
test_images_path = './test_images/'
output_video_file = 'result.mp4'

mtx = None

def load_image(filename):
    return mpimg.imread(filename)

def calibrate_camera(rows=6, cols=9):
    print('start calibration...')
    global mtx
    filenames = glob('./camera_cal/*.jpg')
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # Prepare object points, like (0,0,0), (1,0,0)...
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2) # x, y coordinates

    for f in filenames:
        print('processing ' + f + '...')
        img = load_image(f)
        plt.imshow(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols,rows), None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

        # draw and display the corners
        img = cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
        plt.imshow(img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # save undistorted images
    if ret:
        for f in filenames:
            print('processing ' + f + '...')
            image = load_image(f)
            undist = cv2.undistort(image, mtx, dist, None, mtx)
            mpimg.imsave(output_images_path + 'undistorted_' + f.split('/')[-1], undist, cmap=None)

        # process test_images
        testfilenames = glob('./test_images/*.jpg')
        for f in testfilenames:
            print('processing ' + f + '...')
            img = load_image(f)
            undist = cv2.undistort(img, mtx, dist, None, mtx)
            mpimg.imsave(output_images_path + 'undistorted_' + f.split('/')[-1], undist, cmap=None)

    return mtx, dist

# abs_sobel_thresh function:
# input: image, gradient orientation, threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh=(0,255)):
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# magnitude thresholding
# input: image, sobel_kernel_size, threshold_value
# return: the magnitude of the gradient
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# direction thresholding
# input: image, threshold_range, sobel_kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# return a binary image based on HLS of a given image
def color_threshold(image, s_thresh=(170, 255), sx_thresh=(20, 100)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,2]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= lthresh[0]) & (l_channel <= lthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1) & (l_binary == 1)] = 1
    return output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width=50, window_height=80, margin=100):

    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))
    return window_centroids

def pipeline(img):
    # a thresholded image
    warped = hls_color_binary(img)
    window_centroids = find_window_centroids(warped, window_width=50, window_height=80, margin=100)
    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
	        # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    return output

def image_gen():
    global mtx
    images = glob('./test_images/*.jpg')

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img = cv2.undistort(img,mtx,dist,None,mtx)

        preprocessImage = np.zeros_like(img[:,:,0])
        gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255))
        grady = abs_sobel_thresh(img, orient='y', thresh=(25,255))
        c_binary = color_threshold(img, sthresh=(50,255), vthresh=(100,255), lthresh=(50,255))
        m_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(0,25))
        d_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.0,1.5))
        preprocessImage[((gradx == 1) & (grady == 1) | (m_binary == 1) & (d_binary == 1) & (c_binary == 1))] = 255

        img_size = (img.shape[1], img.shape[0])
        bot_width = 0.76
        mid_width = 0.08
        height_pct = 0.62
        bottom_trim = 0.935
        src = np.float32([[img.shape[1]*(0.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(0.5+mid_width/2),img.shape[0]*height_pct],
            [img.shape[1]*(0.5+bot_width/2),img.shape[0]*bottom_trim], [img.shape[1]*(0.5-bot_width/2),img.shape[0]*bottom_trim]])
        offset = img_size[0]*0.80
        dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(preprocessImage,M,img_size,flags=cv2.INTER_LINEAR)

        window_width = 25
        window_height = 88

        curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = 25, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor = 15)

        window_centroids = curve_centers.find_window_centroids(warped)

        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        rightx = []
        leftx = []

        for level in range(0, len(window_centroids)):
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])

            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)

            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        template = np.array(r_points+l_points,np.uint8)
        zero_channel = np.zeros_like(template)
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        result = cv2.addWeighted(warpage,1,template,0.5,0.0)

        yvals = np.linspace(0, 719, num=720)
        res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)

        left_fit = np.polyfit(res_yvals, leftx, 2)
        left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
        left_fitx = np.array(left_fitx,np.int32)

        right_fit = np.polyfit(res_yvals, rightx, 3)
        right_fitx = right_fit[0]*yvals*yvals*yvals + right_fit[1]*yvals*yvals + right_fit[2]*yvals + right_fit[3]
        righ_fitx = np.array(right_fitx,np.int32)
        left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        middel_marker = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

        road = np.zeros_like(img)
        road_bkg = np.zeros_like(img)

        cv2.fillPoly(road, [left_lane], color=[255,0,0])
        cv2.fillPoly(road, [right_lane], color=[0,0,255])
        cv2.fillPoly(road, [middel_marker], color=[0,255,0])
        cv2.fillPoly(road_bkg, [left_lane], color=[255,255,255])
        cv2.fillPoly(road_bkg, [right_lane], color=[255,255,255])

        road_warped = cv2.warpPerspective(road,Minv,img_size,flags=cv2.INTER_LINEAR)
        road_warped_bkg = cv2.warpPerspective(road_bkg,Minv,img_size,flags=cv2.INTER_LINEAR)

        base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
        result = cv2.addWeighted(base, 1.0, road_warped, 1.0, 0.0)

        ym_per_pix = curve_centers.ym_per_pix
        xm_per_pix = curve_centers.xm_per_pix

        curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
        curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])

        camera_center = (left_fitx[-1] + right_fitx[-1])/2
        center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
        side_pos = 'left'
        if center_diff <= 0:
            side_pos = 'right'

        cv2.putText(result, 'Radius of Curvature = ' +str(round(curverad,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff,3))) +'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

        write_name = output_images_path+'result_'+str(idx)+'.jpg'
        cv2.imwrite(write_name, result)

if __name__ == '__main__':
    global mtx
    mtx, dist = calibrate_camera()
    image_gen()
