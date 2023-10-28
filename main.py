import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import supervision as sv
import streamlit as st
from ultralytics.utils.plotting import Annotator
import requests
import json
def pushbullet_noti(title, body):

    TOKEN = 'o.f2LHazMZNZ66hErEV38RF2m6S3mCitHD'

    msg = {"type": "note", "title": title, "body": body}

    resp = requests.post('https://api.pushbullet.com/v2/pushes',
                         data=json.dumps(msg),
                         headers={'Authorization': 'Bearer ' + TOKEN,
                                  'Content-Type': 'application/json'})
    if resp.status_code != 200:
        raise Exception('Error', resp.status_code)
    else:
        print('Message sent')
model=YOLO('yolov8s.pt')
# source='ganpati.mp4'
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
print(class_list)
# tracker=Tracker()
avg=[]
avg_pred=[]
overall_action=['nonviolence','nonviolence']
st.title('Crowd Detection App')
st.sidebar.title('Crowd Detection Sidebar')
st.sidebar.subheader('DashBoard')

detection_type = st.sidebar.selectbox(
    'Choose the App mode', ['About APP', 'Run on Video','Open Webcam','Suspicious Activity Detection','Cleanliness Detector','Data Insights'])
if detection_type == 'About APP':
    st.markdown(
        'This Application helps in detection of Crowds from  VIDEO or from your WEBCAM depending on your App mode. ')
    st.markdown('''
    About the Author: \n
    Hey this is Syntax Error from SJCEM.\n

    This APP help in the management of crowd in a particular location


    SAMPLE OUTPUT:
    ''')

    # st.video('')

elif detection_type == 'Run on Video':
    avg=[]
    avg_pred=[]

    # cap=cv2.VideoCapture(source)
    st.sidebar.markdown('-----')
    source = st.sidebar.file_uploader(
        "UPLOAD a video", type=['mp4'])
    if source:
        cap=cv2.VideoCapture(f"C:\\tadomal\people-going-in-out-counting\Videos\{source.name}")
        

        avg1=0
        global prediction
        FRAME_WINDOW = st.image([])
        t = st.empty()
        s=st.empty()
        p=st.empty()
        q=st.empty()
        r=st.empty()
        def on_update():
            global prediction
            data = len(detections) 
            t.text(f"Number of persons are {data}")
            if data<10:
                prediction ='less Crowded'
            elif data>10 and data<12:
                prediction='less Crowded'
            elif data>12:
                prediction='Over Crowded'
            s.text(f"Prediction is {prediction}")
            avg_pred.append(prediction)


        # st.text(f'Number of persons present = {avg1}')


        while True:    
            ret,frame = cap.read()
            if not ret:
                break
            frame=cv2.resize(frame,(1020,500))
            results=model.predict(frame)
            detections = sv.Detections.from_ultralytics(results[0])
            a=results[0].boxes.data
            detections = detections[detections.class_id == class_list.index("person")]
            print(len(detections))

            px=pd.DataFrame(a).astype("float")
            for index,row in px.iterrows():
        #        print(row)
                x1=int(row[0])
                y1=int(row[1])
                x2=int(row[2])
                y2=int(row[3])
                d=int(row[5])
                c=class_list[d]
                if 'person' in c:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                    cvzone.putTextRect(frame,f'{c} num {len(detections)}',(x1,y1),1,1)
                    avg.append(len(detections))
                    on_update()
                    q.text(f"Over all person count {round(sum(avg) / len(avg))}")

                
                
            FRAME_WINDOW.image(frame)
                
            # cv2.imshow("RGB", frame)
            if cv2.waitKey(1)&0xFF==27:
                break
        p.text(f"Over all person Count is {round(sum(avg) / len(avg))}")
        r.markdown(f"Over all Prediction is {max(set(avg_pred), key = avg_pred.count)}")

        cap.release()
        cv2.destroyAllWindows()
elif detection_type == 'Open Webcam':
    avg=[]
    avg_pred=[]
    # cap=cv2.VideoCapture(source)
    st.sidebar.markdown('-----')
    
    cap=cv2.VideoCapture(0)
    

    avg1=0
    global prediction1
    FRAME_WINDOW = st.image([])
    t = st.empty()
    s=st.empty()
    p=st.empty()
    q=st.empty()
    r=st.empty()
    def on_update_live():
        global prediction1
        data = len(detections) 
        t.text(f"Number of persons are {data}")
        if data<10:
            prediction1 ='less Crowded'
        elif data>10 and data<12:
            prediction1='less Crowded'
        elif data>12:
            prediction1='Over Crowded'
        s.text(f"Prediction is {prediction1}")
        avg_pred.append(prediction1)


    # st.text(f'Number of persons present = {avg1}')


    while True:    
        ret,frame = cap.read()
        if not ret:
            break
        frame=cv2.resize(frame,(1020,500))
        results=model.predict(frame)
        detections = sv.Detections.from_ultralytics(results[0])
        a=results[0].boxes.data
        detections = detections[detections.class_id == class_list.index("person")]
        print(len(detections))

        px=pd.DataFrame(a).astype("float")
        for index,row in px.iterrows():
    #        print(row)
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            c=class_list[d]
            if 'person' in c:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cvzone.putTextRect(frame,f'{c} num {len(detections)}',(x1,y1),1,1)
                avg.append(len(detections))
                on_update_live()
                q.text(f"Over all person count {round(sum(avg) / len(avg))}")

            
            
        FRAME_WINDOW.image(frame)
            
        # cv2.imshow("RGB", frame)
        if cv2.waitKey(1)&0xFF==27:
            break
    p.text(f"Over all person Count is {round(sum(avg) / len(avg))}")
    r.markdown(f"Over all Prediction is {max(set(avg_pred), key = avg_pred.count)}")

    cap.release()
    cv2.destroyAllWindows()  




elif detection_type == 'Suspicious Activity Detection':
        source=None
        z=st.empty()
        source = st.sidebar.file_uploader(
        "UPLOAD a video", type=['mp4'])
        model=YOLO('besta.pt')
        FRAME_WINDOW = st.image([])
        if source:
            cap=cv2.VideoCapture(f"C:\\tadomal\people-going-in-out-counting\Videos\{source.name}")
        else:
            cap=cv2.VideoCapture(0)
        while True:    
            ret,frame = cap.read()
            if not ret:
                break
            frame=cv2.resize(frame,(1020,500))
            results=model.predict(frame)
            detections = sv.Detections.from_ultralytics(results[0])
            a=results[0].boxes.data
            for r in results:
                
                annotator = Annotator(frame)
                
                boxes = r.boxes
                for box in boxes:
                    
                    b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)])
                    overall_action.append(model.names[int(c)])
                    z.markdown(f"Over all action Prediction is {max(set(overall_action), key = overall_action.count)}")
                    
               
            frame = annotator.result() 
            FRAME_WINDOW.image(frame)

            if cv2.waitKey(1)&0xFF==27:
                    break
        pushbullet_noti(title='suspicious activity detected, prediction is ',body=(max(set(overall_action), key = overall_action.count)))
        cap.release()
        cv2.destroyAllWindows()



elif detection_type == 'Cleanliness Detector':
        source=None
       
        source = st.sidebar.file_uploader(
        "UPLOAD a video", type=['mp4'])
        model=YOLO('best_garbagef.pt')
        FRAME_WINDOW = st.image([])
        if source:
            cap=cv2.VideoCapture(f"C:\\tadomal\people-going-in-out-counting\Videos\{source.name}")
        else:
            cap=cv2.VideoCapture(0)
        while True:    
            ret,frame = cap.read()
            if not ret:
                break
            frame=cv2.resize(frame,(1020,500))
            results=model.predict(frame)
            detections = sv.Detections.from_ultralytics(results[0])
            a=results[0].boxes.data
            for r in results:
                
                annotator = Annotator(frame)
                
                boxes = r.boxes
                for box in boxes:
                    
                    b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    annotator.box_label(b, 'garbage')
                    
                    
                    
                
            frame = annotator.result() 
            FRAME_WINDOW.image(frame)

            if cv2.waitKey(1)&0xFF==27:
                    break
        cap.release()
        cv2.destroyAllWindows()
elif detection_type == 'Data Insights':
    import pandas as pd
    import numpy as np

    # Create a sample dataset
    data = {
        'Train Name': ['Train A', 'Train B', 'Train C', 'Train A', 'Train B', 'Train C'],
        'Platform': [1, 2, 1, 2, 1, 2],
        'Arrival Time': ['08:00', '08:15', '08:10', '09:00', '09:15', '09:10'],
        'Departure Time': ['08:10', '08:25', '08:20', '09:10', '09:25', '09:20'],
        'Passengers on Platform': [50, 60, 45, 70, 55, 75]
    }

    df = pd.DataFrame(data)

    # Convert time columns to datetime
    df['Arrival Time'] = pd.to_datetime(df['Arrival Time'], format='%H:%M')
    df['Departure Time'] = pd.to_datetime(df['Departure Time'], format='%H:%M')

    # Calculate the time spent at the platform
    df['Time Spent (minutes)'] = (df['Departure Time'] - df['Arrival Time']).dt.total_seconds() / 60

    # Display the dataset
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Title of the Streamlit app
    st.title('Exploratory Data Analysis (EDA) for Train Timings and Passengers on Platform')

    # Display the dataset
    st.subheader('Dataset Overview')
    st.write(df)

    # Summary statistics
    st.subheader('Summary Statistics')
    st.write(df.describe())

    # Data Visualization
    st.subheader('Data Visualization')

    # Histogram for passengers on the platform
    st.write('Histogram for Passengers on Platform')
    plt.hist(df['Passengers on Platform'], bins=10, edgecolor='k')
    plt.xlabel('Number of Passengers')
    plt.ylabel('Frequency')
    st.pyplot()

    # Relationship between time spent and passengers
    st.write('Scatter Plot: Time Spent vs. Passengers')
    sns.scatterplot(data=df, x='Time Spent (minutes)', y='Passengers on Platform', hue='Platform')
    plt.xlabel('Time Spent (minutes)')
    plt.ylabel('Number of Passengers')
    st.pyplot()

    # Insights
    st.subheader('Insights')

    # Calculate mean passengers on each platform
    mean_passengers = df.groupby('Platform')['Passengers on Platform'].mean()
    st.write('Mean Passengers on Each Platform:')
    st.write(mean_passengers)

    # You can add more insights and visualizations as needed

    # Footer
    st.text('Your EDA App is ready!')

    # Run the app with 'streamlit run your_script.py'