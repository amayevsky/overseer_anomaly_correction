import enum

import cv2
import numpy

from yolo.utils.network import yolo
from yolo.utils.detector import detect
from yolo.utils.drawer import draw_boxes


class ObservationAttribute:
	def __init__(self, index):
		self.index = index

	def __get__(self, obj, obj_type=None):
		return obj[self.index]

	def __set__(self, obj, value):
		obj[self.index] = value

class YoloObservation:
	x_min = ObservationAttribute(0)
	x_max = ObservationAttribute(1)
	y_min = ObservationAttribute(2)
	y_max = ObservationAttribute(3)
	x_centre = ObservationAttribute(4)
	y_centre = ObservationAttribute(5)
	score = ObservationAttribute(6)
	area = ObservationAttribute(7)

	def __init__(self, data, metadata):
		self.data = list(data)
		x_span = self.x_max - self.x_min
		y_span = self.y_max - self.y_min
		self.data.append(x_span * y_span)
		self.metadata = metadata

	def __getitem__(self, key):
		return self.data[key]

	@classmethod
	def set_metadata(cls, raw_image_size):
		height, width = raw_image_size
		cls.x_min.metadata = (False, 0.0, width)
		cls.x_max.metadata = (False, 0.0, width)
		cls.y_min.metadata = (False, 0.0, height)
		cls.y_max.metadata = (False, 0.0, height)
		cls.x_centre.metadata = (True, 0.0, width)
		cls.y_centre.metadata = (True, 0.0, height)
		cls.score.metadata = (False, 0.0, 1.0)
		cls.area.metadata = (
			True, 0.0, float(numpy.prod(raw_image_size)))


class YoloModel:
	_PROCESSED_IMAGE_SIZE = (416, 416)

	def __init__(self, model_file):
		self._model = yolo()
		self._model.load_weights(model_file)
		self.feature_metadata = {}

	def process_image(self, image, suppress=True):
		processed_image = cv2.resize(image, self._PROCESSED_IMAGE_SIZE)
		boxes, labels = detect(processed_image, self._model)
		processed_image, box_data = draw_boxes(processed_image, boxes, labels)
		if not suppress:
			_report_box_data(processed_image, box_data)
		observations = []
		for box in box_data:
			observation_data = [x[0] for x in box]
			observation = YoloObservation(
				data=observation_data, metadata=self.feature_metadata)
			observations.append(observation)
		return processed_image, observations

	def set_metadata(self, raw_image_size):
		height, width = raw_image_size
		self.feature_metadata = {
			'x_min': (False, 0.0, width),
			'x_max': (False, 0.0, width),
			'y_min': (False, 0.0, height),
			'y_max': (False, 0.0, height),
			'x_centre': (True, 0.0, width),
			'y_centre': (True, 0.0, height),
			'score': (False, 0.0, 1.0),
			'area': (True, 0.0, float(numpy.prod(raw_image_size))),
		}


def _report_box_data(image, box_data):
	for index, box in enumerate(box_data):
		print(
			'x_min:', box[0][0],
			'x_max:', box[1][0],
			'y_min:', box[2][0],
			'y_max:', box[3][0],
			'x_centre:', box[4][0],
			'y_centre:', box[5][0],
			'prob', box[6][0],
		)
		box_image = image[box[2][0]:box[3][0], box[0][0]:box[1][0]]
		if box_image.shape[0] <= 0 and box_image.shape[1] <= 0:
			continue
		while True:
			key = cv2.waitKey(30)
			if key == 27:  # dafuq
				break
			cv2.imshow('Image prediction', image)
			cv2.imshow(str(index), box_image)
		cv2.destroyAllWindows()
