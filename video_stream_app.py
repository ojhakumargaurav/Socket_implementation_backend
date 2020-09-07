from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send
from yolo_video import YoloNetwork
import base64
app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*", logger=True)

# @app.route('/')
# def index():
#     return render_template('index.html')


@socketio.on('pingServer')
def test_message(message):
    print("message")
    emit("pingServer", {'data': message})


@socketio.on('video_frame')
def test_video_streaming(video_file):
    print("video file receiving")
    yn = YoloNetwork()
    final_video_bytes = yn.run_yolo(video_file)
    emit("send_video", final_video_bytes)


# @socketio.on('my broadcast event', namespace='/test')
# def test_message(message):
#     emit('my response', {'data': message['data']}, broadcast=True)


@socketio.on('connect')
def test_connect():
    print("connected to server")
    emit('messageChannel', {'data': 'Connected'})
    print("message send successfuly")


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    YoloNetwork.load_yolo_components()
    socketio.run(app, host='0.0.0.0')
