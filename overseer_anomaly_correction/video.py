import cv2

class Video:
	def __init__(self, data_source):
		self.source = data_source
		self._data = None

	def __enter__(self):
		self._data = cv2.VideoCapture(self.source)
		return self

	def __exit__(self, *args, **kwargs):
		if self._data is not None:
			self._data.release()
			self._data = None

	def __iter__(self):
		return self.generate_frames()

	def generate_frames(self):
		while True:
			return_value, frame = self._data.read()
			if return_value is False:
				return
			if cv2.waitKey(1) & 0xFF == ord('q'):  # wat
				return
			yield frame
