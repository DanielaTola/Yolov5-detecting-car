import cv2
import torch

#importa modelo preentrenado de yolo
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detector(input_souce):
    #comprueba si el ingreso es un video 
    is_video = input_souce.endswith('.mp4')
    

    if is_video: 
        cap = cv2.VideoCapture(input_souce)
        while cap.isOpened():
            status, frame = cap.read()
            if not status:
                break
            detect_and_draw(frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
    else:
        frame = cv2.imread(input_souce)
        detect_and_draw(frame)
        cv2.waitKey(0)
    

def detect_and_draw(frame):
    #inferencia de deteccion
    pred = model(frame)
    #xmin,ymin,xmax,ymax bording box
    df = pred.pandas().xyxy[0]
    #define el nivel de confianza al que hace la deteccion y de la clase que se desea detectar 
    df = df[(df["confidence"] > 0.8) & (df["name"] == "car")]     
    
    #Dibuja los bording box de la deteccion del objeto. 
    for i in range(df.shape[0]):
        bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
        class_name = df.iloc[i]['name']
        confidence = round(df.iloc[i]['confidence'], 2)

        cv2.rectangle(frame, (bbox[0], bbox[1], 
                              bbox[2], bbox[3]), 
                              (255, 0, 0), 1)
        cv2.putText(frame, f"{class_name}: {confidence}", 
                    (bbox[0], bbox[1] - 15), 
                    cv2.FONT_HERSHEY_PLAIN, 2, 
                    (255, 0, 0), 1)

    cv2.imshow("frame",frame)


if __name__ == '__main__':
    detector("data/carros.jpeg")