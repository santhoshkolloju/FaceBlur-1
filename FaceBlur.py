import sys
import os
from os.path import isfile, join

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

import cv2
import face_recognition

class FaceBlur(object):

    def __init__(self, input_dir='images' , max_blurring_iterations=5, result_dir=None):
        """
        Constructor of FaceBlur Object
        :param input_dir: directory containing images to process
        :param max_blurring_iterations: maximal iterations for bluring
        :param result_dir: directory of resulting images
        """

        self._max_blurring_iterations = max_blurring_iterations
        self.input_dir = input_dir
        if result_dir is None:
            self.result_dir=input_dir+"_blurred"
        else:
            self.result_dir = result_dir

    def blur_face(self,image, face_location):
        """
        Bluring face in face_location on image as long as face is visible, or max_iterations are reached
        :param image: to blur out face
        :param face_location: location of face
        :return: image with blurred out area
        """

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        i = 0
        while (not face_recognition.face_locations(image) == []) and i < self._max_blurring_iterations:
            #blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
            blurred = cv2.blur(blurred, (10, 10))
            i += 1
        return blurred,face_location

    def blur_face_helper(self, args):
        """
        Helper function for pooling
        :param args: arugments
        """
        return self.blur_face(*args)

    def detect_and_blur_faces(self, image_name):
        """
        Detect faces and blur them in image_name
        :param image_name: name of image to process
        :return: processed image
        """

        image = cv2.imread(os.path.join(self.input_dir,image_name))
        print(image_name)
        face_locs = face_recognition.face_locations(image)

        list = []
        for face_location in face_locs:

            top, right, bottom, left = face_location
            # face_recognition allowes negative numbers, so here we filter them
            right = max(right, 0)
            right = max(right, 0)
            bottom = max(bottom, 0)
            left = max(left, 0)
            face = image[top:bottom, left:right]
            list.append((face,face_location))

        res_L = []
        with ThreadPoolExecutor() as executor:
            res_L = [executor.submit(self.blur_face_helper, x) for x in list]

            for ft in concurrent.futures.as_completed(res_L):
                i, face_location = ft.result()
                top, right, bottom, left = face_location
                # face_recognition allowes negative numbers, so here we filter them
                right = max(right, 0)
                right = max(right, 0)
                bottom = max(bottom, 0)
                left = max(left, 0)
                image[top:bottom, left:right]=i

        cv2.imwrite(join(self.result_dir, image_name), image)


    def detect_and_blur_helper(self, args):
        """
        Helper function for pooling
        :param args: arugments
        """

        self.detect_and_blur_faces(args)

    def start_processing(self):
        """
        Starts processing
        """

        all_imgs = [f for f in os.listdir(self.input_dir) if isfile(join(self.input_dir, f))]

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        with Pool() as p:
            list = [(img) for img in all_imgs]
            p.map(self.detect_and_blur_helper, list)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        arg = "images/"
    else:
        arg = sys.argv[1]

    fb = FaceBlur(arg)
    fb.start_processing()