from ultralytics import YOLO
import cv2
import json

model = YOLO("/home/hossein/PycharmProjects/yolo-linux/yolo/yolov8n-pose.pt")
results = model.predict(source='/home/hossein/Downloads/man.jpeg',
                        show=False,
                        save=False,
                        device='cuda:0')
# get results
for r in results:
    keypoints = r.tojson()

# converting output(str) into dict
keypoints_json = json.loads(keypoints)  # str into list
# print(type(keypoints_json))
keypoints_dict = {}
for data in keypoints_json:  # list into dict
    for i, (x, y) in enumerate(zip(data['keypoints']['x'], data['keypoints']['y'])):
        keypoints_dict[f'point_{i + 1}'] = {'x': x, 'y': y}
# print(type(keypoints_dict))
print(keypoints_dict['point_1'])

# get image
image = cv2.imread("/home/hossein/Downloads/man (copy).jpeg")

# extract x,y and draw red point on image
for key in keypoints_dict:
    value = keypoints_dict[key]
    x = value['x']
    y = value['y']
    cv2.circle(image,
               (int(x),
                int(y)),
               radius=1,
               color=(0, 0, 255),
               thickness=-1)

cv2.imshow('Image with Keypoints', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
