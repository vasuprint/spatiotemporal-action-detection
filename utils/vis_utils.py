import cv2

def draw_boxes(frame, boxes, actions):
    for box, action in zip(boxes, actions):
        cv2.rectangle(frame, (int(box[0]), int(box[1])), 
                      (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, action, (int(box[0]), int(box[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame