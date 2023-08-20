import torch
import cv2 # opencv2 package for python.


def plot_boxes(results, frame):
    """
    plots boxes and labels on frame.
    :param results: inferences made by model
    :param frame: frame on which to  make the plots
    :return: new frame with boxes and labels plotted.
    """
    labels, names, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
        bgr = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
        label = f"{int(row[4] * 100)}"
        index = int(row[4] * 100)

        cv2.putText(frame, names[index], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Total Targets: {n}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

cap = cv2.VideoCapture(0)
assert cap is not None

names = model.names

while (True):

    # Capture the video frame by frame
    ret, frame = cap.read()

    # Inference
    # results = model([frame])
    results = model(imgs)
    labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()

    # Results
    # results.print()
    # results.save()  # or .show()
    # results.show()

    frame = plot_boxes([labels, names, cord], frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]

