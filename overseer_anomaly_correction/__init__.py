import logging
import yaml
import sys


def run(settings_path='settings.yaml'):
	with open(settings_path) as settings_file:
		settings = yaml.safe_load(settings_file)
	logger = logging.getLogger()
	log_settings = settings.get('log', default={})
	log_level = getattr(logging, log_settings.get('level', default='DEBUG'))
	logger.setLevel(log_level)
	# create a file handler
	log_file = log_settings.get('file')
	if log_file is not None:
		file_handler = logging.FileHandler(log_settings['file'])
		# create a logging format
		formatter = logging.Formatter(
			'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		file_handler.setFormatter(formatter)
		# add the handlers to the logger
		logger.addHandler(file_handler)
		# For printing to stdout
		logger.addHandler(logging.StreamHandler(sys.stdout))
	input_type = settings.get('input_type', default='video_stream')
	model = settings.get('model', default='yolo')
	file_path = settings.get('file_path')
	if file_path:
		output_path = settings.get(
			'output_path', default=f'{input_type}.processed')
	else:
		output_path = None
	logging.info('Input type: %s', input_type)
	logging.info('Model: %s', model)
	logging.info('File path (for video): %s', file_path)
	logging.info('Output path (for video): %s', output_path)
	
	user = SimulatedUser(input_type, model, file_path, output_path)
	user.use_lots()
