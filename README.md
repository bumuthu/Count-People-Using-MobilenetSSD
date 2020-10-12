# Count-People-Using-MobilenetSSD

## Dependancy Installation 

Following python libraries has to be installed on your device in a virtual environment.
```
pip install -r requirements.txt
```
When you install dlib you may have to install cmake as well.
I developed the code using ​Python 3.8​ interpreter. If you have other Python 3 version, I am not
sure that it definitely works. Please try it.

## Get running

Hereby, I have provided optional parameters for tuning as follows.
```
python main.py --skip-frames=30 --confidence=0.4
```
By default these are 30 and 0.4 respectively. You can change confidence in order to tune the
result for your environment. Skip-frames can be increased in order to maximize the speed if
your device would be lagging. But small skip-frames give higher accuracy.
