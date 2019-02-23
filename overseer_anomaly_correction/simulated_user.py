from overseer_anomaly_correction.model import YoloModel
from overseer_anomaly_correction.overseer import Overseer
from overseer_anomaly_correction.video import Video

class SimulatedUser:
	def __init__(self, model):
		self.model = YoloModel('yolo/initial-model.h5')
		self.the_help = Overseer()

	def watch_video(self, video):
		with video:
			for frame in video:
				self.see_image(image=frame)

	def see_image(self, image):
		processed_image, observations = self.model.process_image(image)
		corrected_observations = self.the_help.get_corrections(observations)
