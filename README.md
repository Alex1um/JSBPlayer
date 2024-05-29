# Player bot for Just Shapes & Beats game

## Install

* clone the repository
* install depencies: `pip install -r requirements.txt`

## Running

### On video

In file video.py:

* Edit path to video in cv2.VideoCapture on line 10
* Set the record variable on line 9 to False if no need to record processed frames
* Enable some filters in get_policy function on line 23
* run video.py file

### On game

For linux devices:

* Use any software to capture window and put stream into video device. For example OBS Studio.
* Edit video device number if necessary
* Run kb.py with sudo
* Press enter when level starts and switch to the game
* Close opencv window by pressing q on level end.