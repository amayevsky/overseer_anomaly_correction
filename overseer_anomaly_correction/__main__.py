import argparse
import logging

from overseer_anomaly_correction import run


logging.basicConfig(level=logging.DEBUG)
parser = argparse.ArgumentParser()
run()
