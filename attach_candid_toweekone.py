import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils,rotate_bound
import math
from PIL import Image

import utils
import models
import NonLinearLeastSquares
import ImageProcessing
from drawing import *
import drawing
import FaceRendering

'''
currently the script is very slow for a live video and is not responding, should
check why and change it in such a manner that it can be processed very fast

removing box as its taking a lot of processing power
'''

def drawProjectedShape(img, x, projection, mesh, params, lockedTranslation=False):
    localParams = np.copy(params)

    if lockedTranslation:
        localParams[4] = 100
        localParams[5] = 200

    projectedShape = projection.fun(x, localParams)

    drawPoints(img, projectedShape.T, (0, 0, 255))
    drawMesh(img, projectedShape.T, mesh)
    drawCross(img, params)
    return projectedShape;



def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground (filter)
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
 
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src



def calculate_inclination(point1, point2):
    x1,x2,y1,y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180/math.pi*math.atan((float(y2-y1))/(x2-x1))
    #print ('incline return val'),
    #print(incl);
    return incl

'''
def calculate_boundbox(list_coordinates):
    x = min(list_coordinates[:,0])
    y = min(list_coordinates[:,1])
    w = max(list_coordinates[:,0]) - x
    h = max(list_coordinates[:,1]) - y
    #print ('calculate bound box return val '),
    #print (x,y,w,h);
    return (x,y,w,h)
'''

def get_face_boundbox(points, face_part):
    if face_part == 1:
        (x,y,w,h) = calculate_boundbox(points[17:22]) #left eyebrow
    elif face_part == 2:
        (x,y,w,h) = calculate_boundbox(points[22:27]) #right eyebrow
    elif face_part == 3:
        (x,y,w,h) = calculate_boundbox(points[36:42]) #left eye
    elif face_part == 4:
        (x,y,w,h) = calculate_boundbox(points[42:48]) #right eye
    elif face_part == 5:
        (x,y,w,h) = calculate_boundbox(points[29:36]) #nose
    elif face_part == 6:
        (x,y,w,h) = calculate_boundbox(points[48:68]) #mouth
    #print ('get_face_boundbox return val'),
    #print (x,y,w,h);
    return (x,y,w,h)

'''
def combine_two_color_images_with_anchor(img1, img2, anchor_y, anchor_x):
    foreground, background = img1.copy(), img2.copy()
    # Check if the foreground is inbound with the new coordinates and raise an error if out of bounds
    #foreground is the filter 
    background_height = background.shape[1]
    background_width = background.shape[1]
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    while foreground_height+anchor_y >= background_height or foreground_width+anchor_x >= background_width:
        #resize the filter till it fits 
        foreground = cv2.resize(foreground,dsize=None,fx=0.5,fy=0.5,interpolation = cv2.INTER_CUBIC)
        foreground_height = foreground.shape[0]
        foreground_width = foreground.shape[1]
        #raise ValueError("The foreground image exceeds the background boundaries at this location")
        
    alpha =0.9

    

    # do composite at specified location
    start_y = anchor_y
    start_x = anchor_x
    end_y = anchor_y+foreground_height
    end_x = anchor_x+foreground_width

    print (foreground.shape[:2],(background[start_y:end_y, start_x:end_x,:]).shape[:2],start_y,end_y,start_x,end_x)

    blended_portion = cv2.addWeighted(foreground,
                alpha,
                background[start_y:end_y, start_x:end_x,:],
                1 - alpha,
                0)
    background[start_y:end_y, start_x:end_x,:] = blended_portion
    return background;
    #cv.imshow('composited image', background)

'''

model = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)

#load the candid
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel('candide.npz');
projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])


#video = cv2.VideoCapture(0);
#print('recording in progress');


dog = cv2.imread('glass_2.png',-1); #dog ears filter
print('loaded filter');

modelParams = None
lockedTranslation = False
drawOverlay = True
writer = None


img = cv2.imread('img_1.jpg')
#img = cv2.imread('img_2.jpg')

#the second image taken for the purpose of analysis
second_img = np.copy(img); 

textureImg = cv2.imread('jolie.jpg');
#textureImg = cv2.imread('.../data/beard_1.png',-1);

#while True:
#ret,img = video.read();

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
rects_0 = detector(gray,0)
rows,cols = dog.shape[0], dog.shape[1]



##changes needed to render only the beard or something 
textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
renderer = FaceRendering.FaceRenderer(img, textureImg, textureCoords, mesh)


for i in rects_0:   
    shape = face_utils.shape_to_np(predictor(gray,i))
    #print (shape);

    '''
    for i in range(1,7):                                    #detection of all 6  facial features 
        (x,y,w,h) = get_face_boundbox(shape, i)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1) #draws a rectangle in the image
    '''
    
    
    incl = calculate_inclination(shape[36], shape[42])  #should repeat to all faces 
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2),incl,1)
    dst = cv2.warpAffine(dog,M,(cols,rows))
    dst = rotate_bound(dog, incl)

    #print(dst.shape, dog.shape);
    #cv2.imshow('the rotated dog fiter',dst)
    
    #resize the dst according to the facial perceptions
    x_dist = int(shape[26][0]-shape[17][0]) +10 ;

    #maintains the perspection of the whole image // do we need like this ?
    #y_dist = int(dst.shape[1]*(x_dist/dst.shape[0]));

    #converting the y_dist to allign with the eyes
    eye = shape[36:48];
    y_dist = max(eye[:,1]) - min(eye[:,1]) +20 ;  #min and max cordinates subtracted to get the over all height

     
    dst = cv2.resize(dst,(x_dist,y_dist),interpolation=cv2.INTER_CUBIC)

    #print(dst.shape,(shape[17]),img.shape);
    

    #uncomment this if u wish to see the image with filter added with dlib estimation
    #img = transparentOverlay(img,dst,(shape[17][0],shape[17][1]));

cv2.imshow('after filter',img);
    


        #img = combine_two_color_images_with_anchor(dst,img, shape[17][1], shape[17][0])

### main parts of the face swap starts from here

maxImageSizeForDetection = 320;
shapes2D = utils.getFaceKeypoints(img, detector, predictor, maxImageSizeForDetection)


firsttime = True; #if the rendering is happening for the first time 
if shapes2D is not None:
        for shape2D in shapes2D:
            
            #3D model parameter initialization
            modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

            #3D model parameter optimization
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

            #rendering the model to an image
            shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
            renderedImg = renderer.render(shape3D)

            #blending of the rendered face with the image
            mask = np.copy(renderedImg[:, :, 0])
            renderedImg = ImageProcessing.colorTransfer(img, renderedImg, mask)

            
            if firsttime :
                blend_img = ImageProcessing.blendImages(renderedImg, img, mask)
                firsttime = False;
            else:
                blend_img = ImageProcessing.blendImages(renderedImg,blend_img,mask)
            
            #drawing of the mesh and keypoints
            if drawOverlay:
                drawPoints(img, shape2D.T)

                
                red_dots=drawProjectedShape(img, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)
                red_dots=red_dots.T
                
                
                
                drawPoints(second_img,red_dots,(0,0,255));
                drawNumbers(second_img,red_dots);
                
                
                
                                   

'''
key = cv2.waitKey(1)
if key == 27:
    break
'''

cv2.imshow('overlapped image',blend_img);
cv2.imshow('wireframed image',img);
cv2.imshow('Face boundry',second_img);        
    
#video.release()

#webcam isnt working -- try restarting 
