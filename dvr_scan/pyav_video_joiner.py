import logging
from typing import Optional, Tuple, Union, AnyStr, List

import av
import numpy
from av.filter import Graph
from scenedetect import FrameTimecode
from scenedetect.video_stream import VideoOpenFailure

FRAMERATE_DELTA_TOLERANCE: float = 0.1

logger = logging.getLogger("dvr_scan")


# ----------------------------------------------------
# 1) PyAV 기반 VideoStreamPyAv 클래스 새로 정의
# ----------------------------------------------------
class VideoStreamPyAv:
    """
    PyAV 기반으로 영상 디코딩 및 전처리를 수행하는 클래스.
    VideoStreamCv2와 동일한 인터페이스를 제공하여
    drop-in 교체가 가능하도록 구성.
    """
    def __init__(self, path: Union[AnyStr, List[AnyStr]], scale_factor: int = 1):
        if isinstance(path, (str, bytes)):
            path = [path]
        
        self.paths = path
        self.scale_factor = scale_factor
        self._curr_path_index = 0
        
        # 전체 프레임 수 누적
        self._total_frames = 0
        self.__decode_failures = 0
        
        # 현재 파일 관련 변수들
        self._container = None
        self._video_stream = None
        self._frame_index = 0
        
        # 필터 그래프 관련 변수들
        self._graph = None
        self._buffer_src = None
        self._buffer_sink = None
        
        # 첫 비디오 파일 로드
        self._load_video(self.paths[self._curr_path_index])

    def _load_video(self, path: str):
        """비디오 파일을 로드하고 필터 그래프를 초기화합니다."""
        try:
            self._container = av.open(
                path,
                options={
                    "skip_loop_filter": "all",
                    "threads": "0"
                }
            )
        except Exception as e:
            msg = f"Could not open video via PyAV: {path}"
            raise VideoOpenFailure(msg) from e

        # 비디오 스트림 추출
        self._video_stream = next(
            (s for s in self._container.streams if s.type == 'video'),
            None
        )
        if self._video_stream is None:
            msg = f"No video stream found in file: {path}"
            raise VideoOpenFailure(msg)

        # 첫 프레임 디코딩 시도
        first_frame = None
        try:
            for packet in self._container.demux(self._video_stream):
                for frame in packet.decode():
                    first_frame = frame
                    break
                if first_frame is not None:
                    break
        except av.error.Error as e:
            msg = f"Failed to decode first frame: {path}"
            raise VideoOpenFailure(msg) from e

        if not first_frame:
            msg = f"No frame could be decoded: {path}"
            raise VideoOpenFailure(msg)

        # 메타데이터 추출
        self._time_base = first_frame.time_base
        self._width = first_frame.width
        self._height = first_frame.height
        self._pix_fmt = first_frame.format.name

        sar = getattr(self._video_stream, 'sample_aspect_ratio', None)
        pixel_aspect = f"{sar.numerator}/{sar.denominator}" if sar else "1/1"

        # 프레임레이트 설정
        self._frame_rate = float(self._video_stream.average_rate) \
            if self._video_stream.average_rate else 30.0

        # 현재 파일의 총 프레임 수 계산
        if self._video_stream.duration is not None:
            duration_sec = self._video_stream.duration * self._video_stream.time_base
            curr_total_frames = int(duration_sec * self._frame_rate)
        else:
            curr_total_frames = 0
        
        self._total_frames += curr_total_frames

        # 컨테이너 위치 초기화
        self._container.seek(0)

        # 필터 그래프 초기화
        self._setup_filter_graph(pixel_aspect)

        # 디코딩용 패킷 준비
        self._packets = self._container.demux(self._video_stream)

    def _setup_filter_graph(self, pixel_aspect: str):
        """필터 그래프를 설정합니다."""
        self._graph = Graph()
        
        scale_width = max(1, self._width // self.scale_factor)
        scale_height = max(1, self._height // self.scale_factor)
        
        buffer_args = (
            f"video_size={self._width}x{self._height}:"
            f"pix_fmt={self._pix_fmt}:"
            f"time_base={self._time_base}:"
            f"pixel_aspect={pixel_aspect}"
        )
        
        self._buffer_src = self._graph.add("buffer", buffer_args)
        scale_filter = self._graph.add("scale", f"w={scale_width}:h={scale_height}")
        gblur_filter = self._graph.add("gblur", "sigma=2")
        self._buffer_sink = self._graph.add("buffersink")

        self._buffer_src.link_to(scale_filter)
        scale_filter.link_to(gblur_filter)
        gblur_filter.link_to(self._buffer_sink)

    def read(self, decode: bool = True) -> Optional[numpy.ndarray]:
        """
        다음 프레임을 읽고 반환. 더 이상 없으면 None 반환.
        """
        try:
            for packet in self._packets:
                for frame in packet.decode():
                    if decode:
                        # 필터 그래프 통과
                        self._buffer_src.push(frame)
                        filtered_frame = self._buffer_sink.pull()
                        # RGB to BGR 변환 (OpenCV 호환을 위해)
                        arr = filtered_frame.to_ndarray(format='bgr24')
                    else:
                        # decode=False이면, 단순 카운트만 올리고 패킷 스킵
                        arr = None
                    self._frame_index += 1
                    return arr

            # 현재 파일 끝에 도달했으면 다음 파일 로드 시도
            if (self._curr_path_index + 1) < len(self.paths):
                self._curr_path_index += 1
                self._load_video(self.paths[self._curr_path_index])
                return self.read(decode=decode)

            return None  # False 대신 None 반환

        except Exception:
            self.__decode_failures += 1
            return None  # False 대신 None 반환
        except (av.error.BlockingIOError, av.error.EOFError):
            return None  # False 대신 None 반환

    def seek(self, target: FrameTimecode):
        """
        forward seek만 지원한다고 가정. target 이 현재 프레임보다 미래면,
        그 시점까지 read()로 프레임을 버립니다.
        """
        current_position = self.position
        if target < current_position:
            # PyAV로 정확히 뒤로 가기 어려우므로, 일단 전체 seek(0) 후 재진행
            self._container.seek(0)
            self._packets = self._container.demux(self._video_stream)
            self._frame_index = 0

        while self.position < target:
            frame = self.read(decode=False)
            if frame is False or frame is None:
                break

    @property
    def frame_size(self) -> Tuple[int, int]:
        """(width, height)"""
        return (self._width, self._height)

    @property
    def framerate(self) -> float:
        """비디오 프레임레이트 (frames/sec)"""
        return self._frame_rate

    @property
    def frame_rate(self) -> float:
        """초당 프레임 수"""
        return self._frame_rate

    @property
    def duration(self) -> FrameTimecode:
        """대략적인 총 프레임 수 기반 FrameTimecode 반환."""
        return FrameTimecode(self._total_frames, self._frame_rate)

    @property
    def base_timecode(self) -> FrameTimecode:
        """시작 시점 (항상 0 프레임)"""
        return FrameTimecode(0, self._frame_rate)

    @property
    def position(self) -> FrameTimecode:
        """현재 디코딩된 프레임 인덱스를 기반으로 Timecode 반환."""
        return FrameTimecode(self._frame_index, self._frame_rate)

    @property
    def position_ms(self) -> float:
        """현재 위치(밀리초)"""
        return (float(self._frame_index) / self._frame_rate) * 1000.0

    @property
    def decode_failures(self) -> int:
        """디코딩 실패 횟수를 반환합니다."""
        return self.__decode_failures

    @decode_failures.setter
    def decode_failures(self, value: int):
        """디코딩 실패 횟수를 설정합니다."""
        self.__decode_failures = value

    # _decode_failures property는 이전 코드와의 호환성을 위해 유지
    @property
    def _decode_failures(self) -> int:
        return self.__decode_failures

    @_decode_failures.setter
    def _decode_failures(self, value: int):
        self.__decode_failures = value

    @property
    def resolution(self) -> Tuple[int, int]:
        """비디오 해상도 (width x height)"""
        return self.frame_size

    @property
    def total_frames(self) -> int:
        """전체 프레임 수"""
        return self._total_frames

