import torch
import numpy as np
import cv2
from time import time


class deteksi_objek:

    def __init__(self, capture_index, model_name):
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        membuat video capture
        """

        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        load yolov5 model
        """
        if model_name:
            model = torch.hub.load(
                'ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5',
                                   'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        menerima sebuah frame tunggal (dalam format numpy/list/tuple) sebagai masukan, 
        dan menilai frame tersebut menggunakan model yolo5. 
        Model tersebut dipindahkan ke perangkat yang ditentukan dan frame masukan 
        diterjemahkan menjadi sebuah daftar dengan satu frame. 
        Model yolo5 kemudian memproses frame dan mengeluarkan label dan koordinat dari objek yang terdeteksi di frame. 
        Fungsi ini mengembalikan label dan koordinat tersebut sebagai sebuah tuple.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def detect_object(self, image_path):
        """
        menerima sebuah path ke file citra sebagai input 
        dan akan mengembalikan label objek dan koordinat bounding box yang didetekso oleh model Yolov5 pada citra tersebut
        """
        image = cv2.imread(image_path)
        image = cv2.resize(image, (416, 416))
        return self.score_frame(image)

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        fungsi untuk membuat kotak
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(
                    row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(
                    labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        looping untuk frame video
        """
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:

            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (416, 416))

            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            # print(f"Frames Per Second : {fps}")

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv5 Detection', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()


# Create a new object and execute.
detector = deteksi_objek(capture_index=0, model_name='best.pt')
detector()
# results = detector.detect_object('iee/53.jpg')
# labels, cord = results
# print("Labels:", labels)
# print("Coordinates:", cord)

# # Create an image copy
# image = cv2.imread('iee/53.jpg')
# image_copy = image.copy()

# # Plot boxes on the image copy
# image_copy = detector.plot_boxes(results, image_copy)

# # Show the image with boxes
# cv2.imshow('YOLOv5 Detection', image_copy)
# cv2.waitKey(0)

# jika objek luar rumah harus ada/diluar dataset harus ada outputnya apa "dengan ditambahkan bondinbbox"


# 3 bisa
# 5 sebagian
