import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        

def get_img_obj_points():
    """
    Using chessboard to find te image points and object points.
    """
    images = glob.glob("camera_cal/calibration*.jpg")
    objpoints = []
    imgpoints = []
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    
    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret==True:
            imgpoints.append(corners)
            objpoints.append(objp)
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
    
    return (imgpoints, objpoints)


def undistort(img, objpoints, imgpoints):
    """
    Method to undistort img using the output from get_img_obj_points
    """
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    undist = np.uint8(255 * undist)
    return undist


def hls_thresh(img, thresh=(0, 255)):
    """
    Isolate s channel from hls colorspace.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel>=thresh[0]) & (s_channel<=thresh[1])] = 1
    
    return binary_output


def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    '''
    Produce thresholded sobel image.
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient=='y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take absolute value of the sobel derivative
    abs_sobel = np.absolute(sobel)
    # Scale it back to range(0, 255)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """
    Produce sobel image thresholded by gradient magnitude.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelX**2, sobelY**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = np.uint8((gradmag/scale_factor))
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Produce binary image based on gradient direction.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelX = np.absolute(sobelX)
    abs_sobelY = np.absolute(sobelY)
    # Calculate direction of the gradient
    dir_sobel = np.arctan2(abs_sobelX, abs_sobelY)
    
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel>=thresh[0]) & (dir_sobel<=thresh[1])] = 1
    
    return binary_output


def region_of_interest(img, vertices):
    """
    Mask the image that most likely contains lane line.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def warp(img):
    """
    Perform perspective transform. Output will be a bird-view of the lane line.
    """
    img_size = (img.shape[1], img.shape[0])
    # Top left, Top right, bottom left, bottom right
    src = np.float32(
            [[585, 457],
             [700, 457],
             [200, 719],
             [1125, 719]])
    dst = np.float32(
            [[300, 0],
             [960, 0],
             [300, 719],
             [960, 719]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    scaled_warped = np.uint8(255*warped/np.max(warped))
    
    return scaled_warped


def unwarp(img):
    """
    Unwarp a warped image. (eg. unwarp an bird-view lane line image back to original.)
    """
    img_size = (img.shape[1], img.shape[0])
    # Top left, Top right, bottom left, bottom right
    src = np.float32(
            [[585, 457],
             [700, 457],
             [200, 719],
             [1125, 719]])
    dst = np.float32(
            [[300, 0],
             [960, 0],
             [300, 719],
             [960, 719]])
    
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    
    return unwarped


def find_curvature(left_x, left_y, right_x, right_y):
    """
    Find the curvature for both left and right lane.
    """
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit_cr = np.polyfit(left_y*ym_per_pix, left_x*xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_y*ym_per_pix, right_x*xm_per_pix, 2)
    y_eval = np.max(left_y)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    print(left_curverad, right_curverad)
    
    return (left_curverad, right_curverad)


def draw(warped, left_fitx, right_fitx, ploty, img):
    """
    Draw a polygon presenting the lane line area back to original video frame.
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    newwarp = unwarp(color_warp)

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    
    return result


def sliding_window(warped):
    """
    Using sliding window to find the lane line segments. This method will be used at the start of a video or when there is no       lane line found on the previous video frame.
    """
    img_height, img_width = warped.shape[0], warped.shape[1]
    histogram = np.sum(warped[img_height//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped, warped, warped))
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows                                  (How to choose this???)
    n_windows = 9
    # Set height of windows
    window_height = img_height // n_windows
    # Identify the x and y positions of all nonzero pixels in the image. 
    nonzero = warped.nonzero() # This returns indices of nonzero value.
    X_nonzero = nonzero[1]
    Y_nonzero = nonzero[0]
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    min_pixel = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_height - (window + 1) * window_height
        win_y_high = img_height - (window) * window_height
        win_x_left_low = leftx_current - margin
        win_x_left_high = leftx_current + margin
        win_x_right_low = rightx_current - margin
        win_x_right_high = rightx_current + margin
        cv2.rectangle(out_img, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (0, 255, 0), 2)
        
        # Identify the nonzero pixels in x and y within the window
        left_lane_ind = ((Y_nonzero >= win_y_low) & (Y_nonzero < win_y_high) & \
                         (X_nonzero >= win_x_left_low) &  (X_nonzero < win_x_left_high)).nonzero()[0]
        
        right_lane_ind = ((Y_nonzero >= win_y_low) & (Y_nonzero < win_y_high) & \
                         (X_nonzero >= win_x_right_low) &  (X_nonzero < win_x_right_high)).nonzero()[0]
        left_lane_inds.append(left_lane_ind)
        right_lane_inds.append(right_lane_ind)
        # If you found > minpix pixels, recenter next window on their mean position
        if(len(left_lane_ind) > min_pixel):
            leftx_current = np.int(np.mean(X_nonzero[left_lane_ind]))
        if(len(right_lane_ind) > min_pixel):
            rightx_current = np.int(np.mean(X_nonzero[right_lane_ind]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    left_x = X_nonzero[left_lane_inds]
    left_y = Y_nonzero[left_lane_inds]
    right_x = X_nonzero[right_lane_inds]
    right_y = Y_nonzero[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_poly_para = np.polyfit(left_y, left_x, 2)
    right_poly_para = np.polyfit(right_y, right_x, 2)
    
    # Find curvature
    curvature = find_curvature(left_x, left_y, right_x, right_y)
    
    # Visualisation
    # Draw the polynomial line
    plot_y = np.linspace(0, img_height-1, img_height)
    left_poly = left_poly_para[0] * plot_y**2 + left_poly_para[1] * plot_y + left_poly_para[2]
    right_poly = right_poly_para[0] * plot_y**2 + right_poly_para[1] * plot_y + right_poly_para[2]
    #Color the laneline
    out_img[Y_nonzero[left_lane_inds], X_nonzero[left_lane_inds]] = [255, 0, 0]
    out_img[Y_nonzero[right_lane_inds], X_nonzero[right_lane_inds]] = [0, 0, 255]
    
    return (out_img, left_poly, right_poly, plot_y, left_poly_para, right_poly_para, curvature)


def not_sliding_window(warped, left_poly_para, right_poly_para):
    """
    Method to find the lane line when there was a fit on the previous video frame.
    """
    img_height, img_width = warped.shape[0], warped.shape[1]
    out_img = np.dstack((warped, warped, warped))
    window_img = np.zeros_like(out_img)
    nonzero = warped.nonzero()
    X_nonzero = nonzero[1]
    Y_nonzero = nonzero[0]
    margin = 100
    left_lane_inds = ((X_nonzero > left_poly_para[0]*(Y_nonzero**2) + left_poly_para[1]*Y_nonzero \
                       + left_poly_para[2] - margin )) & ((X_nonzero < left_poly_para[0]*(Y_nonzero**2)
                       + left_poly_para[1]*Y_nonzero + left_poly_para[2] + margin ))
    right_lane_inds = ((X_nonzero > right_poly_para[0]*(Y_nonzero**2) + right_poly_para[1]*Y_nonzero \
                       + right_poly_para[2] - margin )) & ((X_nonzero < right_poly_para[0]*(Y_nonzero**2)
                       +right_poly_para[1]*Y_nonzero + right_poly_para[2] + margin ))
    
    left_x = X_nonzero[left_lane_inds]
    left_y = Y_nonzero[left_lane_inds]
    right_x = X_nonzero[right_lane_inds]
    right_y = Y_nonzero[right_lane_inds]
    left_poly_para = np.polyfit(left_y, left_x, 2)
    right_poly_para = np.polyfit(right_y, right_x, 2)
    plot_y = np.linspace(0, img_height-1, img_height)
    left_poly = left_poly_para[0] * plot_y**2 + left_poly_para[1] * plot_y + left_poly_para[2]
    right_poly = right_poly_para[0] * plot_y**2 + right_poly_para[1] * plot_y + right_poly_para[2]
    
    # Find curvature
    curvatures = find_curvature(left_x, left_y, right_x, right_y)
    
    # Drawing
    out_img[Y_nonzero[left_lane_inds], X_nonzero[left_lane_inds]] = [255, 0, 0]
    out_img[Y_nonzero[right_lane_inds], X_nonzero[right_lane_inds]] = [0, 0, 255]
    # Generate a polygon to illustrate the search window area
    left_line_window1 = np.array([np.transpose(np.vstack([left_poly-margin, plot_y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_poly+margin, plot_y])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_poly-margin, plot_y]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_poly+margin, plot_y])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return (out_img, left_poly, right_poly, plot_y, left_poly_para, right_poly_para, curvatures)


def pipeline(img, objpoints, imgpoints, left_lines, right_lines, sanity_fail_count): 
    """
    Pipleline to be used in video.
    """
    ksize = 3
    # Undistort the image
    undist_img = undistort(img, objpoints, imgpoints)

    # Calculate: x sobel, y sobel, magnitute of sobel, direction of sobel
    gradx = abs_sobel_thresh(undist_img, orient='x', sobel_kernel=ksize, thresh=(20, 200))
    grady = abs_sobel_thresh(undist_img, orient='y', sobel_kernel=ksize, thresh=(20, 200))
    mag_binary = mag_thresh(undist_img, sobel_kernel=ksize, thresh=(10, 255))
    dir_binary = dir_threshold(undist_img, sobel_kernel=ksize, thresh=(0, np.pi/2))
    s_binary = hls_thresh(undist_img, thresh=(140, 255))

    # Combined the previous results
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | (s_binary == 1)] = 1

    # Mask the image
    imshape = combined.shape
    vertices = np.array([[(0, imshape[0]),(0.51*imshape[1], 0.55*imshape[0]), (0.49*imshape[1], 0.55*imshape[0]), 
                              (imshape[1], imshape[0])]], dtype=np.int32)
    masked_img = region_of_interest(combined, vertices)

    # Warp the image
    warped = warp(masked_img)
    
    use_slide = (len(left_lines) == 0) or (len(right_lines) == 0) \
                or (not (left_lines[-1].detected and right_lines[-1].detected)) \
                or (sanity_fail_count[0] > 4)
    if use_slide: # If in the last frame, no line was detected or is the beginning or a video.
        #Sliding window   
        print("Using sliding window")
        (out_img, left_poly, right_poly, plot_y, left_poly_para, right_poly_para, curvatures) = sliding_window(warped)
        sanity_fail_count[0] = 0
    else:
        print("Using non-sliding window")
        # No-sliding window
        left_poly_para = left_lines[-1].current_fit
        right_poly_para = right_lines[-1].current_fit
        (result, left_poly, right_poly, plot_y, left_poly_para, right_poly_para, curvatures) = not_sliding_window(warped,                                                                                                left_poly_para, right_poly_para)
        # If this line detection does not pass the sanity_check
        if len(left_lines) > 0:
            last_left_line = left_lines[-1]
            last_right_line = right_lines[-1]
        if not sanity_check(last_left_line, last_right_line, curvatures):
            print("Sanity Fail")
            sanity_fail_count[0] += 1
            left_poly = last_left_line.allx
            right_poly = last_right_line.allx
            plot_y = last_left_line.ally
            curvatures = (last_left_line.radius_of_curvature, last_right_line.radius_of_curvature)
            left_poly_para = last_left_line.current_fit
            right_poly_para = last_right_line.current_fit
        else:
            sanity_fail_count[0] = 0

    left_line = Line()
    right_line = Line()
    left_line.allx = left_poly
    right_line.allx = right_poly
    left_line.ally = plot_y
    right_line.ally = plot_y
    left_line.current_fit = left_poly_para
    right_line.current_fit = right_poly_para
    left_line.detected = True
    right_line.detected = True
    left_line.radius_of_curvature = curvatures[0]
    right_line.radius_of_curvature = curvatures[1]
    left_lines.append(left_line)
    right_lines.append(right_line)
    # Draw green polygon on the image.
    final = draw(warped, left_poly, right_poly, plot_y, img)
    
    return final


def sanity_check(last_left_line, last_right_line, curvatures):
    previous_curvature = last_left_line.radius_of_curvature
    left_curvature = curvatures[0]
    right_curvature = curvatures[1]
    if ((left_curvature - right_curvature) < 200) and ((left_curvature - previous_curvature) < 800):
        return True
    else:
        return False
    