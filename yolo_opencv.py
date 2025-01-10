#-------------------------------------------------------------# 

 #### ####  :*-_-*-_-*-_-*- -  
########### | [ minecraft mob ai ] - test model with cv 
 #########  |    -> code credit - http://www.arunponnusamy.com
   #####    |                                                      
     #      :*-_-*-_-*-_-*- -
   
    #| useful explanations of what's happening in code: 
    #| https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9

#-------------------------------------------------------------#



#-------------------------------------------------------------#
# |
# |   ## ##   :
# |  #######  | [ imports ]
# |   #####   |    -> needs pip, numpy & opencv installed
# |     #     :
# |
# :

import cv2
import argparse
import numpy as np
#-------------------------------------------------------------#


#-------------------------------------------------------------#
# |
# |           :
# |   ## ##   | [ arguments ]
# |  #######  |    -> this section creates a custom terminal
# |   #####   |          command to run the script.
# |     #     |    -> maybe change layout to be easier to type
# |           :
# |
# :

ap = argparse.ArgumentParser(description='Object detection script.')
ap.add_argument('-i', '--image', required = True, 
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True, 
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True, 
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required = True, 
                help = 'path to text file containing class names')
args = ap.parse_args()

# :
# |
# |   ### ###             
# |  #########           
# |   ## [ section notes ]
# |     ### | > $ python yolo_opencv.py --i [path to img]
# |      #  |   --config [path to .yaml] --weights [ path 
# |         |   to .weights] --classes [ path to .txt]
# |         :
# |
#-------------------------------------------------------------#



#-------------------------------------------------------------#
# |
# |           :
# |   ## ##   | [ output layers + bounding box ]
# |  #######  |    -> getting output layer names from archite-
# |   #####   |          cture 
# |     #     |    -> drawing box around detected objects
# |           :
# |
# :

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x,y,x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# :
# |
# |   ### ###             
# |  #########           
# |   ## [ section notes ]
# |     ### | > get_output_layers - retrieve names of the out-
# |      #  |     put layers from cv2.dnn
# |         |        1) layer_names = net.getLayerNames(): con-
# |         |           tains the IDs of all layers in nn in
# |         |           the order they appear
# |         |        2) try-except block: in try block, used i-1
# |         |           because getLayerNames() starts at 1, but
# |         |           python lists start at 0. in except block,
# |         |           throw exception --> assumes net.getUncon-
# |         |           nectedOutLayers() returns a list of lists.
# |         |           retrieves output layer names using i[0]-1
# |         |           to handle formatting
# |         | 
# |         | > draw_prediction - draw rectangle around objects
# |         |     using predicted coords, displays the class label
# |         |     & color codes boxes
# |         |        1) parameters:
# |         |             a) img - image to analyze
# |         |             b) class_id - index of predicted class
# |         |             c) confidence - prediction score b/w 1 
# |         |                and 0
# |         |             d) x, y - coords of top-left of rect
# |         |             e) x_plus_w, y_plus_h - coords of bot-
# |         |                right of rect
# |         |        2) label = str(classes[class_id]): converts
# |         |           id to the object. classes contains all
# |         |           classes the model detects. 
# |         |        3) color = COLORS[class_id]: retrieves color
# |         |           from COLORS corresponding to object's class.
# |         |        4) cv2.rectangle([...]): draws rectangle around
# |         |           detected object
# |         |        5) cv2.outText([...]): displays class label
# |         :
# |
#-------------------------------------------------------------#



#-------------------------------------------------------------#
# |
# |
# |   ## ##  :
# |  ####### | [ preparing input ]
# |   #####  |    -> pip must be installed
# |     #    :
# |
# |
# :

image = cv2.imread(args.image)
if image is None:
    print("error, unable to load image")
    exit()

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

# :
# |
# |   ### ###             
# |  #########           
# |   ## [ section notes ]
# |     ### | > loads image, extracts dimensions, processes it
# |      #  |     with YOLO. generates random colors for each
# |         |     class [fix]. resizes image to 416x416
# |         | 
# |         | > loading image {145-148}
# |         |        1) cv2.imread(args.image): reads image
# |         |           from file path. if not found, return
# |         |           none.
# |         |        2) check for none: if image is none, pri-
# |         |           nt error message and exits program
# |         | 
# |         | > getting image dimensions {150-151}
# |         |        1) image.shape: returns dimensions of im-
# |         |           age (height, width, channels)
# |         |        2) image.shape[#]: 0 - height (# of rows
# |         |           of px), 1 - width (# of columns of px),
# |         | 
# |         | > setting up normalization scale {152}
# |         |        1) scale: YOLO uses scale factor of 1/225
# |         |           (0.00392) to normalize px values to 
# |         |           range [0, 1].
# |         |         **(normalization -> converts each px val
# |         |           from normal range [0,255] to [0,1])
# |         | 
# |         | > loading class labels {154-157}
# |         |        1) args.classes: path to file containing
# |         |           class identification labels
# |         |        2) f.readlines() & line.strip() reads the
# |         |           file & removes all whitespace 
# |         |        3) classes: list of strings that represe-
# |         |           nt the labels
# |         | 
# |         | > generating random colors {159} 
# |         |     (not exactly necessary for the code)
# |         |        1) np.random.uniform([...]): generates 
# |         |           random color for each class in an ar-
# |         |           ray of RGB values
# |         |        2) len(classes): # of classes detected
# |         |           by YOLO model -> works sort of like a
# |         |           for loop where it goes through the l- 
# |         |           ist & assigns the values
# |         |        3) size=(len(classes), 3): makes sure all
# |         |           classes has a corresponding RGB
# |         | 
# |         | > loading YOLO model {161}
# |         |        1) cv2.dnn.readNet(): loads YOLO model us-
# |         |           ing the config file and weights
# |         |        2) args.config: path to YOLO config files
# |         |        3) args.weights: path to YOLO weight files
# |         |        4) net: opencv dnn model
# |         | 
# |         | > prep image for input to model {163}
# |         |        1) cv2.dnn.blobFromImage(): converts input
# |         |           image into 'blob' to be used as input
# |         |           into the nn (blob is a 'preprocessed
# |         |           format')
# |         |        > image - og image to be processed
# |         |        > scale - normalization factor
# |         |        > (416,416) - resized image dimensions
# |         |        > (0,0,0) - not sure??
# |         |        > True - also not sure what this does
# |         |        > crop=False - makes sure image won't be
# |         |          cropped
# |         | 
# |         | > setting blob as input to model {165}
# |         |        1) net.setInput(blob): feeds image into t-
# |         |           he YOLO model (net) to be processed
# |         | 
# |         :        
# |
#-------------------------------------------------------------#


# 2025-01-10
#-------------------------------------------------------------#
outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height) 
            w = int(detection[2] * Width)
            h = int(detection[3]*Height)
            x = center_x - w/2
            y = center_y - h/2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
#-------------------------------------------------------------#



#-------------------------------------------------------------#
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

if len(indices) > 0:
    for i in indices.flatten():  
        box = boxes[i]
        x, y, w, h = box
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

cv2.imshow("Object Detection", image)
cv2.waitKey()

cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
#-------------------------------------------------------------#