# from model import stupid_model
from model.model1 import big_object_detection
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os
  
import img_convert_binary as binary_convert
import matplotlib.pyplot as plt
from processing import xml_handle

INPUT_PATH = "D:/Data learning/Document Layout Analysis/Challenge 2_ Document Layout Analysis/0825_DataSamples 1/CI0002.png"
# OUTPUT_PATH

if __name__ == '__main__':
    im = cv2.imread(INPUT_PATH)
    im_name = os.path.basename(INPUT_PATH)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    predict = big_object_detection.run(im_rgb)
    plt.imshow(predict)
    plt.show()

    result = xml_handle.get_xml_string(predict, im_name)

    output_file = open('result/' + im_name[:-4] + '.xml', 'w')
    output_file.write(result)
    output_file.close()

    cv2.imwrite('result/' + im_name, im)
    cv2.imwrite('result/' + im_name[:-4] + '.tif', binary_convert.get_binary_image(im))



### For debug
# cv2.imshow('image', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


