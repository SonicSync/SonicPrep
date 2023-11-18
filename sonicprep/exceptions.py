class SonicPrepException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class IOError(SonicPrepException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ReadAudioError(IOError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class AudioTypeError(ReadAudioError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
