![DVR-Scan Logo](https://raw.githubusercontent.com/Breakthrough/DVR-Scan/main/docs/assets/dvr-scan-logo.png)

:vhs: Find and extract motion events in videos.

------------------------------------------------

### Beta Release: v1.7-dev1 (February 2, 2025)

The latest beta of DVR-Scan includes a new GUI.  See [the release page](https://github.com/Breakthrough/DVR-Scan/releases/tag/v1.7-dev1) for download links and screenshots.  Feedback is most welcome (see issue [#198](https://github.com/Breakthrough/DVR-Scan/issues/198)).

![GUI Beta Screenshot](https://github.com/user-attachments/assets/6d6bb509-f40c-48ba-95d9-a7912935e9eb)

------------------------------------------------

### Latest Release: v1.6.2 (December 17, 2024)

**Website**: [dvr-scan.com](https://www.dvr-scan.com)

**User Guide**: [dvr-scan.com/guide](https://www.dvr-scan.com/guide/)

**Documentation**: [dvr-scan.com/docs](https://www.dvr-scan.com/docs/)

**Discord**: [https://discord.gg/UtE6mMSA](https://discord.gg/UtE6mMSA)

------------------------------------------------------

DVR-Scan is a command-line application that **automatically detects motion events in video files** (e.g. security camera footage).  DVR-Scan looks for areas in footage containing motion, and saves each event to a separate video clip.  DVR-Scan is free and open-source software, and works on Windows, Linux, and Mac.

## Quick Install

    pip install dvr-scan[opencv] --upgrade

Windows builds are also available on [the Downloads page](https://www.dvr-scan.com/download/).

## Quickstart

Scan `video.mp4` (separate clips for each event):

    dvr-scan -i video.mp4

Select a region to scan using [the region editor](https://www.dvr-scan.com/guide/):

    dvr-scan -i video.mp4 -r

<img alt="example of region editor" src="https://raw.githubusercontent.com/Breakthrough/DVR-Scan/releases/1.6/docs/assets/region-editor-mask.jpg" width="480"/>

Select a region to scan using command line (list of points as X Y):

    dvr-scan -i video.mp4 -a 50 50 100 50 100 100 100 50

Draw boxes around motion:

    dvr-scan -i video.mp4 -bb

<img alt="example of bounding boxes" src="https://raw.githubusercontent.com/Breakthrough/DVR-Scan/releases/1.6/docs/assets/bounding-box.gif" width="480"/>

Use `ffmpeg` to extract events:

    dvr-scan -i video.mp4 -m ffmpeg

See [the documentation](https://www.dvr-scan.com/docs) for a complete list of all command-line and configuration file options which can be set. You can also type `dvr-scan --help` for an overview of command line options. Some program options can also be set [using a config file](https://www.dvr-scan.com/docs/#config-file).

------------------------------------------------

Copyright © 2016-2024 Brandon Castellano. All rights reserved.
Licensed under BSD 2-Clause (see the LICENSE file for details).
