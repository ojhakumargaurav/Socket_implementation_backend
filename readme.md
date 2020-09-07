# to run this project in docker 

docker build . -t <app_name>
docker run -p <port_no>:8081 <app_name>

# Socket implementation along with Yolo object detection

This project will recieve video frames in blob format via socket from frontend and run yolo detection and 
return the video with the boundaries and object label with probabilities around them.

frontend for the project can be found in Vue js at below mention repo URL:

https://github.com/ojhakumargaurav/Vue-Socket-and-camera-implementation

download the yolo-coco weights and cfg files and names file from following google drive:

https://drive.google.com/drive/folders/12QHspKJ1uWp-A8hAP7l_-b-2QWcysiL7?usp=sharing