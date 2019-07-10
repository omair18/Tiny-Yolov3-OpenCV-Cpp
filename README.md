### Results of Tiny-YOLOv3

<img src = "https://github.com/gulshan-mittal/learnopencv/blob/dev1/ObjectDetection-YOLO/bird_yolo_out_py.jpg" width = 400 height = 300/>


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

