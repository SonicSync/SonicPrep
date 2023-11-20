import os
from pedalboard import *
import random
import itertools
from .io import find_all_files


class Augmenter:

    def __init__(self, **kwargs):
        self.vars = kwargs.get('vars', 3)
        self.ir_dir = kwargs.get('ir_dir', os.path.abspath('impulses'))
        self.chorus = Chorus()
        self.phaser = Phaser()
        self.clipping = Clipping()
        self.compressor = Compressor()
        self.gain = Gain()
        self.limiter = Limiter()
        self.convolution = Convolution()
        self.delay = Delay()
        self.pitch_shift = PitchShift()
        self.resample = Resample()
        self.bitcrush = Bitcrush()

        self.transformations = [self.apply_chorus, self.apply_phaser, self.apply_clipping, self.apply_compression, self.apply_gain,
                                self.apply_limiting, self.apply_convolution, self.apply_delay, self.apply_pitch_shift, self.apply_resampling, self.apply_bitcrush]

        self.combinations = [itertools.combinations(
            self.transformations, r) for r in range(1, len(self.transformations))]

    def apply_chorus(self, audio_data):
        self.chorus.rate_hz = random.uniform(0.5, 20000)
        self.chorus.depth = random.uniform(0, 1)
        self.chorus.centre_delay_ms = random.uniform(0.5, 1000)
        self.chorus.feedback = random.uniform(0, 1)
        self.chorus.mix = random.uniform(0, 1)
        return self.chorus.process(audio_data)

    def apply_phaser(self, audio_data):
        self.phaser.rate_hz = random.uniform(0.5, 20000)
        self.phaser.depth = random.uniform(0, 1)
        self.phaser.centre_frequency_hz = random.uniform(0, 1000)
        self.phaser.feedback = random.uniform(0, 1)
        self.phaser.mix = random.uniform(0, 1)
        return self.phaser.process(audio_data)

    def apply_clipping(self, audio_data):
        self.clipping.threshold_db = random.uniform(-60, 0)
        return self.clipping.process(audio_data)

    def apply_compression(self, audio_data):
        self.compressor.threshold_db = random.uniform(-60, 0)
        self.compressor.ratio = random.uniform(1, 10)
        self.compressor.attack_ms = random.uniform(1, 100)
        self.compressor.release_ms = random.uniform(1, 100)
        return self.compressor.process(audio_data)

    def apply_gain(self, audio_data):
        self.gain.gain_db = random.uniform(-100, 20)
        return self.gain.process(audio_data)

    def apply_limiting(self, audio_data):
        self.limiter.threshold_db = random.uniform(-60, 0)
        self.limiter.release_ms = random.uniform(1, 100)
        return self.limiter.process(audio_data)

    def apply_convolution(self, audio_data):
        self.convolution.mix = random.uniform(0.1, 1)
        self.convolution.impulse_response_filename = random.choice(
            self.ir_files)
        return self.convolution.process(audio_data)

    def apply_delay(self, audio_data):
        self.delay.delay_seconds = random.uniform(0.1, 1000)
        self.delay.feedback = random.uniform(0.1, 1)
        self.delay.mix = random.uniform(0.1, 1)
        return self.delay.process(audio_data)

    def apply_pitch_shift(self, audio_data):
        self.pitch_shift.semitones = random.uniform(-2, 2)
        return self.pitch_shift.process(audio_data)

    def apply_resampling(self, audio_data):
        self.resample.target_sample_rate = random.uniform(400, 120000)
        return self.resample.process(audio_data)

    def apply_bitcrush(self, audio_data):
        self.bitcrush.bit_depth = random.uniform(1, 64)
        return self.bitcrush.process(audio_data)

    def apply_transformations(self, audio_data, combination):
        for transformation in combination:
            audio_data = transformation(audio_data)
        return audio_data

    def apply_combinations(self, audio_data):
        return [self.apply_transformations(audio_data, combination) for combination in self.combinations for _ in self.vars]
