# Face Analysis

Face Detection & Expression & Recognition & Blinking & Drowsiness Detection

## Face Registration

If you want using the face recognition, you mush enroll the face images firstly!

First, you should add the face infomations in the ```./res/faces/faces.yaml```, like blow:

***NOTE:***

you just can set one image for a face now! But I will be upgrade the code later , after the upgrade you can set many images for a face!

```yaml
- face: face1  # faceid
  name: 张晓民  # face name
  images:
    - zhangxiaomin/zhangxiaomin-1.jpg # image path relative to faces.yaml file
```

## Run

### Create a virtualenv

````bash
virtualenv venv

# activate the virtualenv
source venv/bin/activate # macos or linux
venv/Scripts/activate # windows only
````

### Install the dependencies

```bash
pip install -r requirements.txt
```

### Run scripts

```bash
python demo.py
```

## TODO

- [ ] Add new moudle for adding many images for enrollment faces
- [ ] Using the Optical Flow method to stable the face box and face landmarks
- [ ] Show all detected drowsiness face
- [ ] Using Face Tracking instead of Face Detectiong for speed up performance
