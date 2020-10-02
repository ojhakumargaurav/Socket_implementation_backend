from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send, join_room, leave_room
from utils.yolo_video import YoloNetwork
from utils.face_recognise_utils import RecognizeFace
import base64
app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*",
                    logger=True, ping_timeout=2000, ping_interval=2000)
SocketIO()

# @app.route('/')
# def index():
#     return render_template('index.html')


@socketio.on('pingServer')
def test_message(message):
    print("message")
    emit("pingServer", {'data': message})


@socketio.on('run_yolo')
def run_pre_trained_yolo(video_file_with_user_details):
    yn = YoloNetwork()
    final_video_bytes = yn.process_video(
        video_file_with_user_details["video_frame"])
    emit("processed_video", final_video_bytes,
         room=video_file_with_user_details["username"])


@socketio.on('recognize_face')
def recognize_face_in_video(video_file_with_user_details):
    print("video file receiving")
    fr = RecognizeFace()
    final_video_bytes = fr.process_video(
        video_file_with_user_details["video_frame"])
    emit("processed_video", final_video_bytes,
         room=video_file_with_user_details["username"])


@socketio.on('connect')
def test_connect():
    emit('messageChannel', {'data': 'Connected'})
    print("message send successfuly")


@socketio.on("register_user")
def register_user(user_details):
    join_room(user_details["username"])
    print("joined room")


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    YoloNetwork.load_yolo_components()
    socketio.run(app, host='0.0.0.0')
