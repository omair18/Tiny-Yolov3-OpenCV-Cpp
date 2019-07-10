### Tiny-YOLOv3

OpenCV >= 3.4.5

<img src = "https://github.com/omair18/Tiny-Yolov3-OpenCV-Cpp/blob/master/data/yolo-img.png" width = 960 height = 600/>


### Download the Models

Download weights and cfg files from darknet website

### How to run the code


Command line usage for object detection using Tiny-YOLOv3 

* C++:
  * Build
	```bash
	mkdir build
	cd build
	cmake ../
	make
       ```

  * A single image:
        

    ```bash
    ./detector --image=bird.jpg
    ```

    

  * A video file:

    ```bash
     ./detector --video=run.mp4
    ```


* Python

  * A single image:
    	

    ```bash
    python3 object_detection_yolo.py --image=bird.jpg
    ```

  * A video file:

       ```bash
       python3 object_detection_yolo.py --video=run.mp4
       ```

