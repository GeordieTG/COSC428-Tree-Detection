from ultralytics import YOLO
import cv2

def main():
    img_path = 'street.jpeg'

    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 480))

    model = YOLO('yolov8s.pt')
    result = model.predict(img, conf=0.25)[0]
    result_img = result.plot()

    cv2.imshow('result', result_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()