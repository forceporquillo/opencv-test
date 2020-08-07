import cv2 as cv
import numpy as np
import face_recognition as fr

image_train = fr.load_image_file("faces/elon_musk_tesla_3036.jpg")
image_train = cv.cvtColor(image_train, cv.COLOR_BGR2RGB)
image_test = fr.load_image_file("faces/elon_musk_test.jpg")
image_test = cv.cvtColor(image_test, cv.COLOR_BGR2RGB)

print('Original size:', image_train.shape, image_test.shape)
# scale elon training image
scale_percent = 60

width_train = int(image_train.shape[1] * scale_percent / 100)
height_train = int(image_train.shape[0] * scale_percent / 100)
train_dimens = (width_train, height_train)

width_test = int(image_test.shape[1] * scale_percent / 100)
height_test = int(image_test.shape[0] * scale_percent / 100)
test_dimens = (width_test, height_test)

resize_train = cv.resize(image_train, train_dimens, interpolation=cv.INTER_AREA)
resize_test = cv.resize(image_test, test_dimens, interpolation=cv.INTER_AREA)

small_train = cv.resize(resize_train, (0, 0), fx=0.5, fy=0.5)
small_test = cv.resize(resize_test, (0, 0), fx=0.3, fy=0.3)

print('Resized size:', image_train.shape, image_test.shape)
print('small scaled size:', small_train.shape, small_test.shape)

face_location_test = fr.face_locations(resize_train)[0]
encode_elon_musk = fr.face_encodings(resize_train)[0]
cv.rectangle(image_train, (face_location_test[3], face_location_test[0]),
             (face_location_test[1], face_location_test[2]), (255, 0, 255), 2)

face_location_train = fr.face_locations(resize_test)[0]
encode_elon_musk_test = fr.face_encodings(resize_test)[0]
cv.rectangle(image_train, (face_location_train[3], face_location_train[0]),
             (face_location_train[1], face_location_train[2]), (255, 0, 255), 2)

results = fr.compare_faces([encode_elon_musk], encode_elon_musk_test)
face_distance = fr.face_distance([encode_elon_musk], encode_elon_musk_test)
print('Face recognition test', results, face_distance)

cv.putText(small_test, f'{results} {round(face_distance[0], 2)}', (50, 50),
           cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),  2)

cv.imshow("Training: Elon Musk", small_train)
cv.imshow("Test: Elon Test", small_test)
cv.waitKey(0)
