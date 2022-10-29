import numpy as np
import cv2
from PIL import Image

def find_homography_matrix(src_pts, dst_pts):

    A = np.empty((8,8)) # create an empty matrix A of size 8x8
    b = [] # create an empty list to store the values of destination points

    # next, we need to fill the matrix A with the values from the given points
    for i in range(4):
        x_i = src_pts[i][0] # x_i = x coordinate of ith point in source image
        y_i = src_pts[i][1] # y_i = y coordinate of ith point in source image
        x_i_prime = dst_pts[i][0] # x_i_prime = x coordinate of ith dst point
        y_i_prime = dst_pts[i][1] # y_i_prime = y coordinate of ith dst point
        A[2*i] = [x_i, y_i, 1, 0, 0, 0, -x_i*x_i_prime, -y_i*x_i_prime] # update matrix A
        A[2*i+1] = [0, 0, 0, x_i, y_i, 1, -x_i*y_i_prime, -y_i*y_i_prime] # update matrix A
        b.append(x_i_prime)
        b.append(y_i_prime)

    # using least square estimation, we can find the solution to the matrix Ah = b
    # we use the Moore-Penrose pseudoinverse to find the solution
    h = np.matmul(A.T, A)
    h = np.linalg.inv(h)
    h = np.matmul(h, A.T)
    h = np.matmul(h, b) # finishing here, we get the Moore-Penrose pseudoinverse h = (A.T * A)^-1 * A.T * b
    h = np.append(h, 1) # use 1 as the last element of h
    H = np.reshape(h, (3,3)) # reshape h to a 3x3 matrix
    
    # Alternatively, we can use the code below and get the same result
    # h = np.linalg.lstsq(A, b, rcond=0)[0] 
    # h = np.concatenate((h, [1]), axis=0)  
    # H = h.reshape(3,3) 

    return H

def select_src_pts(img):

    img_copy = img.copy() # copy the image
    
    src_pts = [] # create an empty list to store the points
  
    def mouse_callback(event, x, y, flags, param):
        # store the points clicked by the user
        if event == cv2.EVENT_LBUTTONDOWN  and len(src_pts) < 4:
            src_pts.append([x, y])
            
            cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1) # draw a circle on the image to guide the user

            # connect the points with a line
            if len(src_pts) > 1:
                cv2.line(img_copy, tuple(src_pts[-1]), tuple(src_pts[-2]), (0, 255, 0), 1)
            if len(src_pts) == 4:
                cv2.line(img_copy, tuple(src_pts[-1]), tuple(src_pts[0]), (0, 255, 0), 1)
            
            cv2.imshow('Image', img_copy)

    # next, we create a window for user interaction
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)
    cv2.imshow('Image', img_copy)

    # wait for the user to select 4 points
    while len(src_pts) < 4:
        cv2.waitKey(20)
  
    # after selecting 4 poitns, user is given the option to reselect the points,
    # next code gives the instructions to the user        
    cv2.rectangle(img_copy, (0, 0), (275, 65), (255, 255, 255), -1)    
    cv2.putText(img_copy, 'Press r to select again', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img_copy, 'Press any other key to continue', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)    

    # show the instructions
    cv2.imshow('Image', img_copy)

    # if the user wants to reselect, we call the function again. else, we continue
    if cv2.waitKey(0) == ord('r'):
        src_pts = []
        src_pts = select_src_pts(img)
    else:
        cv2.destroyWindow('Image')

    # finally, sort src_pts in this order: bottom left, top left, top right, bottom right
    src_pts = sorted(src_pts, key=lambda x: x[0])
    src_pts[0:2] = sorted(src_pts[0:2], key=lambda x: x[1], reverse=True)
    src_pts[2:4] = sorted(src_pts[2:4], key=lambda x: x[1])

    return src_pts

def get_destination_points(src_pts):

    [top_right_src, top_left_src, bottom_left_src, bottom_right_src] = src_pts # read source points

    # get maximum vertical length of the object (i.e., maximum distance between top and bottom points)
    vlen_max = int(max(np.sqrt((bottom_left_src[0] - top_left_src[0])**2 + (bottom_left_src[1] - top_left_src[1])**2),
              np.sqrt((bottom_right_src[0] - top_right_src[0])**2 + (bottom_right_src[1] - top_right_src[1])**2)))

    # get maximum horizontal length of the object (i.e., maximum distance between left and right points)
    hlen_max = int(max(np.sqrt((top_left_src[0] - top_right_src[0])**2 + (top_left_src[1] - top_right_src[1])**2),
                np.sqrt((bottom_left_src[0] - bottom_right_src[0])**2 + (bottom_left_src[1] - bottom_right_src[1])**2)))

    # get the destination points using the maximum vertical and horizontal lengths
    top_right_dst = [0,hlen_max-1] # bottom left point of the destination image
    top_left_dst = [0,0] # top left point of the destination image
    bottom_left_dst = [vlen_max-1,0] # top right point of the destination image
    bottom_right_dst = [vlen_max-1,hlen_max-1] # bottom right point of the destination image

    dst = [top_right_dst, top_left_dst, bottom_left_dst, bottom_right_dst] # collect dst points in a list
    
    return [dst, hlen_max, vlen_max]

if __name__ == '__main__':
    
    img_file = input('Enter image file name/path: ') # get the image file name from the user
    img = cv2.imread(img_file) # read the image

    src = select_src_pts(img)
    [dst, hlen_max, vlen_max] = get_destination_points(src)

    H = find_homography_matrix(src, dst)
    img_mapped = cv2.warpPerspective(img, H, (vlen_max, hlen_max)) # map the image to the destination image using the homography matrix

    cv2.imshow('Projected Image', img_mapped)
    cv2.waitKey(0)
