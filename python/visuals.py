# visuals.py

import os
from pathlib import Path
import json

from collections import namedtuple
import csv
import cv2 

import matplotlib.pyplot as plt      

# ----------------------------------------------------------------------------
# global declarations 
from global_parameters import UNDEFINED, WordArea

def plot_image(image, word_areas=[], entities=[], graph={}, negative_edges=True):
    """
    display an image, with word bounding boxes and word tags
    """
    
    # deep copy the image before editing it
    img = image.copy()
    
    color_box = (255, 0, 0)  # Blue in BGR
    color_tag = (0, 0, 255)  # Red in BGR
    color_arc = (0, 255, 0)  # Green in BGR
        
    # params for putText
    fontThickness = 1
    fontScale = 0.5

    for wa, entity in zip(word_areas, entities): 
        
        # draw bounding box
        cv2.rectangle(img, (wa.left, wa.top), (wa.right, wa.bottom),(255,0,0),2)
        
        # show tag
        if entity != UNDEFINED:
            _ = cv2.putText(img, entity, (wa.left, wa.top-2), cv2.FONT_HERSHEY_SIMPLEX,  
                fontScale, color_tag, fontThickness, cv2.LINE_AA) 
        
        # draw edge
        if graph.get(wa.idx, UNDEFINED) != UNDEFINED:
            for direction, (neighbour_idx, rd) in graph[wa.idx].items():
                if neighbour_idx != UNDEFINED:
                    neigh = word_areas[neighbour_idx]
                    if negative_edges:
                        if direction == "left":
                            start_point = (neigh.right, (neigh.top + neigh.bottom)//2)
                            end_point = (wa.left, (wa.top + wa.bottom)//2)
                        elif direction == "top":
                            start_point = ((neigh.left + neigh.right)//2, neigh.bottom)
                            end_point = ((wa.left + wa.right)//2, wa.top)
                        else:
                            continue
                    else:
                        if direction == "right":
                            start_point = (neigh.left, (neigh.top + neigh.bottom)//2)
                            end_point = (wa.right, (wa.top + wa.bottom)//2)
                        elif direction == "bottom":
                            start_point = ((neigh.left + neigh.right)//2, neigh.top)
                            end_point = ((wa.left + wa.right)//2, wa.bottom)
                        else:
                            continue                        
                    # draw the edge line
                    image = cv2.line(img, start_point, end_point, color_arc, thickness=2)
                    # print the rd value
                    if abs(rd) > 0.01:
                        middle_point = tuple((x0+x1)//2 for x0, x1 in zip(start_point, end_point))
                        _ = cv2.putText(img, "{}".format(rd), middle_point, cv2.FONT_HERSHEY_SIMPLEX,  
                            fontScale, color_arc, fontThickness, cv2.LINE_AA) 


    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # display the image
    plt.figure(figsize = (10,20))
    plt.imshow(cv_rgb)
    # plt.show()

