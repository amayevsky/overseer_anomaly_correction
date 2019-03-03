import argparse
import itertools
import logging
import math
import sys
import yaml

import cv2
from yolo.utils.network import yolo
from yolo.utils.detector import detect
from yolo.utils.drawer import draw_boxes


class Video:
	def __init__(self, data_source):
		self.source = data_source
		self._data = None
		self._frames = None

	def __enter__(self):
		self._data = cv2.VideoCapture(self.source)
		self._frames = self.generate_frames()
		return self

	def __exit__(self, *args, **kwargs):
		if self._data is not None:
			self._data.release()
			self._data = None

	def __iter__(self):
		return self

	def __next__(self):
		return next(self._frames)

	def generate_frames(self):
		while True:
			return_value, frame = self._data.read()
			if return_value is False:
				return
			if cv2.waitKey(1) & 0xFF == ord('q'):  # wat
				return
			yield frame


class Histories:
	"""
	Assigns identities and tracks historical data
	"""
	def __init__(self, history_duration=300):
		self.identified_histories = {}
		self.history_duration = history_duration

	@property
	def next_identity(self):
		return max(self.identified_histories, default=-1) + 1
	
	def update(self, anonymous_entities):
		available_identities = itertools.count(self.next_identity)
		hypotheses = _configurations(
			anonymous_entities, self.identified_histories)
		snapshot = self.get_snapshot()
		best_fit = min(
			hypotheses,
			key=lambda x: _test_configuration(x, snapshot),
			default={},
		)
		updated_histories = {}
		for entity, identity in best_fit.items():
			if identity is None:
				identity = next(available_identities)
			history = self.identified_histories.get(identity, [])
			history.append(entity)
			updated_histories[identity] = history
			if len(history) > self.history_duration:
				history.pop(0)
		self.histories = updated_histories

	def get_snapshot(self):
		return {x: y[-1] for x, y in self.identified_histories.items()}


class YoloEntity:
	def __init__(self, box):
		self.box = box

	def __sub__(self, subtrahend):
		return (
			math.sqrt(
				math.pow(self.box.x - subtrahend.box.x, 2)
				+ math.pow(self.box.y - subtrahend.box.y, 2)
			)
			+ abs(
				(self.box.w * self.box.h)
				- (subtrahend.box.w * subtrahend.box.h)
			)
		)


class YoloModel:
	PROCESSED_IMAGE_SIZE = (416, 416)

	def __init__(self, model_file, shape):
		self.model = yolo()
		self.model.load_weights(model_file)
		self.shape = shape

	def get_entities(self, image):
		resized_image = cv2.resize(image, self.PROCESSED_IMAGE_SIZE)
		boxes, labels = detect(resized_image, self.model)
		processed_image, _ = draw_boxes(image, boxes, labels)
		entities = [YoloEntity(box=x) for x in boxes]
		return processed_image, entities


def _configurations(entities, identities):
	if not entities:
		return
	if not identities:
		yield {x: None for x in entities}
		return
	entity, *entities = entities
	for identity in identities:
		remaining_identities = [x for x in identities if x != identity]
		for configuration in _configurations(entities, remaining_identities):
			yield {entity: identity, **configuration}


def _test_configuration(configuration, snapshot):
		quality = 0
		for entity, identity in configuration.items():
			if identity is None:
				continue
			quality += entity - snapshot[identity]
		return quality


def frames_and_entities(video_path, model_path):
	with Video(video_path) as video:
		first_frame = next(video)
		model = YoloModel(model_path, first_frame.shape)
		histories = Histories()
		for frame in video:
			unidentified_frame, yolo_entities = model.get_entities(
				frame)
			histories.update(yolo_entities)
			snapshot = histories.get_snapshot()
			identities = list(snapshot)
			boxes = [x.box for x in snapshot.values()]
			identified_frame, _ = draw_boxes(
				frame, boxes, identities)
			yield (identified_frame, unidentified_frame, frame), snapshot


def run(video_path, model_path):
	for frames, _ in frames_and_entities(video_path, model_path):
		identified_frame, unidentified_frame, frame = frames
		cv2.imshow('raw', frame)
		cv2.imshow('unidentified', unidentified_frame)
		cv2.imshow('identified', identified_frame)


def _main():
	parser = argparse.ArgumentParser()
	parser.add_argument('video', help='path to video input')
	parser.add_argument('model', help='path to YOLO model')
	parser.add_argument(
		'--settings', default='settings.yaml', help='path to settings file')
	parser.add_argument('--log-level')
	parser.add_argument('--log-file')
	arguments = parser.parse_args(sys.argv[1:])
	try:
		with open(arguments.settings) as settings_file:
			settings = yaml.safe_load(arguments.settings)
	except FileNotFoundError:
		settings = {}
	log_settings = settings.setdefault('log', {})
	log_level = arguments.log_level
	if not log_level:
		log_level = log_settings.setdefault('level', 'DEBUG')
	logger = logging.getLogger()
	logger.setLevel(log_level)
	log_file = arguments.log_file
	if not log_file:
		log_file = log_settings.get('file')
	if log_file:
		# create a file handler
		file_handler = logging.FileHandler(log_file)
		file_handler.setLevel(log_level)
		# create a logging format
		formatter = logging.Formatter(
			'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		file_handler.setFormatter(formatter)
		# add the handlers to the logger
		logger.addHandler(file_handler)
	logger.addHandler(logging.StreamHandler(sys.stdout))
	run(arguments.video, arguments.model)


if __name__ == '__main__':
	_main()
