import weakref

import numpy


class History:
	def __init__(
			self,
			initial_observation,
			prediction_window,
			anomaly_threshold=2,
			memory_duration=300,
			null_observation_tolerance=5,
			prediction_strategy='sgolay_filter',
	):
		self.metadata = initial_observation.metadata
		self.predictor = prediction_strategy
		self.prediction_window = prediction_window
		self.memory_duration = memory_duration
		self.observations = [initial_observation]
		self.anomaly_map = weakref.WeakKeyDictionary(
			{initial_observation: False})
		self.anomaly_threshold = anomaly_threshold
		self.time_series = self.observations.copy()
		self.null_observation_count = 0
		self.null_obsevation_max = null_observation_tolerance

	def add_observation(self, observation):
		self.observations.append(observation)
		if len(self.observations) > self.memory_duration:
			self.observations.pop(0)
		is_anomalous = self.latest_is_anomalous
		self.anomaly_map[observation] = is_anomalous
		if not is_anomalous:
			return
		

	@property
	def latest_is_anomalous(self):
		while True:
			if self.anomaly_map[self.observations[-1]]:
				return True
			non_anomalies = [
				x for x in self.observations if not self.anomaly_map[x]]
			second_differential = numpy.diff(non_anomalies, n=2)
			if not second_differential:
				return False
			standard_deviation = numpy.std(second_differential)
			for observation, delta in zip(
					non_anomalies[2:], second_differential):
				if abs(delta) > standard_deviation * self.anomaly_threshold:
					self.anomaly_map[observation] = True
					break
			else:
				return False