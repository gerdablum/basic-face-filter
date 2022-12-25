# Real time face filter 

This project realizes a very basic face filter, similar to the snapchat/instagram filters out there (but much more basic). Per default, your face is replaced by hide-the-pain-harold's face. Custom images can be added of course. See the stunning result below:

<img src="example.jpg" width="200"/>

The underlying techniques of this mini-project are face-detection library from [MediaPipe](https://github.com/google/mediapipe), Gooogle's live data and streaming media ML project.
With openCV and some basic numpy operations, we perform the replacement of the face.


```python
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import numpy as np
import random
```

### Detecting faces

For the face recognition, we don't have to do much, as we use the face regognition model of mediapipe.

The ``detection`` method uses the model from google's face detection and returns a result.
We need to convert the image to RGB first (openCV loads the images as BGR for some reason)
To save some memory, we set writeable to false, while processing the image.

The result usually contains detections for each face and we can extract keypoints for eyes, nose and mouth as well as acces the raw data (we will need that later for the bounding box).

The `draw_face_detection` method is used only for debug reasons. It visualises the bounding box of a face and the "keypoints", such as eyes and nose. It's really interesting to see what the library detects, but for the end result we don't need it.


```python
def detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_face_detection(image, results, mp_drawing):
    if results.detections is None:
        return
    for detection in results.detections:
        mp_drawing.draw_detection(image, detection)

```

### Adding the filter

In this part, we actually have to do a little work. In `add_face_overlays` we loop over detections. The result object contains a detection for each detected face. If there is no face detection in the image we do an early return. Then we call the actual face overlay method with one face image per detection.

The `add_face_overlay_method` performs the following steps:

1. create an alphatransparent overlay image with the same size as the original camera frame
2. extract the upper left corner point of the face recognition bounding box as well as the width
3. scale face image to the right width of the detected face
4. place face image at the right position of overlay image
5. combine overlay image and original camera frame to one image and return it

**1. create an alphatransparent overlay** \
We create an empty image (all zeros) which has 4 channels, the 4th is the transparency information. The image is fully transparent. We will place our face image inside this empty overlay. We could save some memory and computation time, if we add the face image directly, but I found it much easier to combine the face with the background image, if they both have the same size

**2. Extract face detection points** \
We have to access the raw data of the detection object, beause we want the upper left corner of the face bounding box to anchor our face image. The raw data coordinates are relative numbers between 0 and 1 so we scale them up to the image width and height. Last we test out and add some values on top to fit the face overlay to the actual face.

**3. Scale face image** \
We want that our face image has always the same size as the original face. Therefore we take the widh of the face detection bounding box (plus some threshold) and scale the face image accordingly. We must maintain the width and height properties. The scaling is done in `scale_image`. The width of the face image is very sensitive to small changes. This could be improved by adding a threshold for size changes.

**4. Place face image at the right position** \
We use the `x_min` and `y_min` coordinates to replace the empty overlay with the face image starting at this x and y position.
Note that the overlay image has the same dimesions as the camera image, whilst the face images has the same size as the detected face bounding box. To prevent out-of-bound errors, we use a smaller cut-out porition of the face image if we are at the borders (x and y smaller that 0 or size of face image overlaps camera image). That makes the code a little messy in this part.

**5. Combine images** \
Now we have two images: One is the camera image, called `background`, the other one is the face images put into the empty overlay, called `foreground_with_alpha`. They both have the same size. \
In `combine_images` we perform the last step, returning them together as one image, with the face image on top and underlaying the camera image. To do this, we create an alpha mask out of the foreground image. This alpha masks contains 0 where the image is transparent, and ones where it is not. \
Our goal is to have `background` cut out where the face image should be positioned, and `foreground_with_alpha` everywhere else except where the actual face is. We realize that by simply multiplying each color channel with the alpha mask and the inverted alpha mask respectively. \
Combining both images by adding them, gives us the correct overlay.


```python
def add_face_overlays(image, face_images, result):
    if result.detections is None:
        return image

    for idx, detection in enumerate(result.detections):
        face_img_no = idx % len(face_images)
        image = add_face_overlay(image, face_images[face_img_no], detection)
    return image

def combine_images(background, foreground_with_alpha):
    output_image_per_channel = 0
    alpha_mask = np.divide(foreground_with_alpha[:, :, 3], 255).astype(bool)
    image_channels = []
    combined_image = []

    for channel in range(background.shape[2]):
        cut_out_background = np.multiply(
            background[:, :, channel], np.invert(alpha_mask))
        cut_out_foreground = np.multiply(
            foreground_with_alpha[:, :, channel], alpha_mask)
        image_channels.append(cut_out_background + cut_out_foreground)

    return (np.dstack(image_channels)).astype(np.uint8)


def scale_image(image, desired_width):

    width, height = image.shape[1], image.shape[0]
    ratio = desired_width / width
    dimension = (int(width * ratio), int(height * ratio))
    return cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)


def add_face_overlay(camera_image, face_img, detection):

    # 1. create a second transparent image for the overlay so we can place harold
    trans_overlay_image = np.zeros((camera_image.shape[0], camera_image.shape[1], 4))

    # 2. extract the upper left corner point of the face recognition bounding box as well as the width.
    face_width = int(
        detection.location_data.relative_bounding_box.width * camera_image.shape[1]) + 10
    x_min = int((1 - detection.location_data.relative_bounding_box.xmin)
                * camera_image.shape[1] - face_width) + 10
    y_min = int(
        detection.location_data.relative_bounding_box.ymin * camera_image.shape[0]) - 40

    # 3. scale face image to the right width of the detected face
    face_img = scale_image(face_img, face_width)

    width_end = (face_img.shape[1]+x_min)
    height_end = (face_img.shape[0]+y_min)

    # handle replacing when at right and lower border
    if width_end > camera_image.shape[1]:
        width_end = camera_image.shape[1]
    if height_end > camera_image.shape[0]:
        height_end = camera_image.shape[0]

    # handle replacing if we are at left and upper border
    start_y = y_min
    start_x = x_min
    start_x_face = 0
    start_y_face = 0
    if x_min < 0:
        start_x = 0
        start_x_face = abs(x_min)
    if y_min < 0:
        start_y = 0
        start_y_face = abs(y_min)

    # 4. plug in face image at right position in overlay image
    trans_overlay_image[start_y:height_end, start_x:width_end, :] = face_img[start_y_face:height_end +
                                                                             start_y_face - start_y, start_x_face:width_end+start_x_face - start_x, :]
    # 5. combine overlay image and original camera frame to one image and return it                                                                             
    return combine_images(camera_image, trans_overlay_image)

```

## Putting it all together

Now we can execute and test our script. We initialize the mediapipe face detection and open a camera stream with our webcam usign this snippet: `cv2.VideoCapture(0)`. If your camera does not open, try a different channel. \
Then in a while loop, we first run the face recognition detection, add the overlay and then show each frame using `cv2.imshow`. End the camera stream by pressing **q**.
feel free to add more face overlays. They have to be transparent but are not restricted to a certain shape. Simply load them at the beginning and add them to the list.


```python

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(0)

# read overlay images
overlay_img_list = []
overlay_img = cv2.imread("harold.png", cv2.IMREAD_UNCHANGED)
overlay_img = np.fliplr(overlay_img)
overlay_img_list.append(overlay_img)

try:
    with  mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            rel, frame = cap.read()
            image, result = detection(frame, face_detection)
            # draw_face_detection(image, result, mp_drawing)
            # mirror the image to make it appear more natural in webcam
            image =  np.fliplr(image)

            image = add_face_overlays(image, overlay_img_list, result)
            

            cv2.imshow('OpenCV Frame', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()

```

## Further improvements

This face filter is, as stated, very very basic. There are some improvements which I might add some day, but propably never...
* Detect rotation of the face and rotate filter accordingly
* Make the size of the filter a little less sensitive to small face width changes
* Sometimes, face detection detects nothing for a short period. Make overlay less sensitive to this as well
* Use more keypoints, like eyes and nose to match the face overlay keypoints with the detected keypoints.
