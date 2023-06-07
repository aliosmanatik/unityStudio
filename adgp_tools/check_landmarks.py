# Metadata fit checking tool for Academic Dataset by Generated Photos by Ali Osman Atik
# 'generated.photos' and 'generated.photos_metadata' folders should be in same directory
# Use "q" for closing application
# Use "d" for deleting image and json metadata
# Use argument -a for auto mode
# Use argument -s for silent auto mode
# Use argument -slm for saving 5 point landmarks
# Use argument -d for delay time on auto mode

import os
import argparse
import json
import cv2
import send2trash


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--photos", required=True, help="Enter the 'generated.photos' folder path ie: 'check_landmarks.py -p generated.photos'")
# ap.add_argument("-m", "--metadata", required=True, help="Enter the metadata folder path")
ap.add_argument("-a", "--auto", action="store_true", default=False, help="Turn auto mode on. If not used manual mode is active Deletes image and json file with features out of range")
ap.add_argument("-s", "--silent", action="store_true", default=False, help="Turn silent auto mode on. Simply hides render windows to speed-up auto mode")
ap.add_argument("-lm5p", "--landmarks5point", action="store_true", default=False, help="Creates 5 point landmarks for images in separate folder")
ap.add_argument("-d", "--delay", default=500, type=int, help="Delay time on auto mode, default 500 ms")

args = vars(ap.parse_args())

photo_path = args["photos"]
# meta_path = args["metadata"]
auto = args["auto"]
silent_auto = args["silent"]
landmarks5point = args["landmarks5point"]
delay = args["delay"]

im_path = [os.path.join(photo_path, i) for i in sorted(os.listdir(photo_path)) if i.endswith('png') or i.endswith('jpg')]
lm_path = [i.replace('png', 'json').replace('jpg', 'json') for
           i in im_path]
lm_path = [os.path.join(photo_path + '_metadata', i.split(os.path.sep)[-1]) for i in lm_path]


# scaled point for 2x up-scaling to 512*512
def point(x, y):
    return int(round(x)/2), int(round(y)/2)


# find mean point for eye region
def mean(region):
    x = y = 0
    for xy in range(len(region)):
        x += region[xy]['x']
        y += region[xy]['y']
    x = x / len(region)
    y = y / len(region)
    return point(x, y)


print("\n ***********************************************************")
print(" * ADGP manual check tool by Ali Osman Atik                *")
print(" * Use any key to advance next image                       *")
print(" * Use 'd' key for deleting current image and json file    *")
print(" * Use 'q' key to exit tool                                *")
print(" * Use 'y' key to confirm, any key to skip                 *")
print(" ***********************************************************\n")

# Margins for data, outliers will be printed in red
pitchMin = -12
pitchMax = 18
rollMax = 10
yawMax = 18
bugMax = 0.2
ageMin = 15

counter = 0

# Info setup on image
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
colorRed = (255, 0, 0)
colorGreen = (0, 255, 0)
thickness = 1

for i in range(len(im_path)):
    counter += 1

    json_file = open(lm_path[i], "r")
    data = json.load(json_file)
    json_file.close()

    pitchData = data['faceAttributes']['headPose']['pitch']
    rollData = data['faceAttributes']['headPose']['roll']
    yawData = data['faceAttributes']['headPose']['yaw']
    bugData = data['bug_probability']
    ageData = data['faceAttributes']['age']

    pitchColor = colorGreen if pitchMin <= pitchData <= pitchMax else colorRed
    rollColor = colorGreen if abs(rollData) <= rollMax else colorRed
    yawColor = colorGreen if abs(yawData) <= yawMax else colorRed
    bugColor = colorGreen if bugData <= bugMax else colorRed
    ageColor = colorGreen if ageData >= ageMin else colorRed

    warning = colorRed in (pitchColor, colorGreen, rollColor, yawColor, bugColor, ageColor)

    # Console info
    im_name = im_path[i].split(os.path.sep)[-1]
    print(' Image ', counter, ' : ', im_name)

    if not silent_auto:
        img = cv2.cvtColor(cv2.imread(im_path[i]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)  # INTER_LINEAR INTER_AREA INTER_CUBIC INTER_LANCZOS4
        img5 = img.copy()

        landmarks = data['faceLandmarks']
        for key in landmarks.keys():
            for p in landmarks[key]:
                cv2.circle(img, point(p['x'], p['y']), 2, (255, 155, 255), 2)

        # Print 5-point facial landmarks
        left_eye = mean(landmarks['left_eye'])
        right_eye = mean(landmarks['right_eye'])
        nose = point(landmarks['nose'][3]['x'], landmarks['nose'][3]['y'])
        left_mouth = point(landmarks['mouth'][0]['x'], landmarks['mouth'][0]['y'])
        right_mouth = point(landmarks['mouth'][6]['x'], landmarks['mouth'][6]['y'])

        cv2.circle(img5, left_eye, 2, (155, 155, 255), 2)
        cv2.circle(img5, right_eye, 2, (155, 155, 255), 2)
        cv2.circle(img5, nose, 2, (155, 155, 255), 2)
        cv2.circle(img5, left_mouth, 2, (155, 155, 255), 2)
        cv2.circle(img5, right_mouth, 2, (155, 155, 255), 2)

        # Render windows
        cv2.putText(img, "Pitch : " + str(pitchData), (20, 30), font, fontScale, pitchColor, thickness, cv2.LINE_AA)
        cv2.putText(img, "Roll  : " + str(rollData), (20, 50), font, fontScale, rollColor, thickness, cv2.LINE_AA)
        cv2.putText(img, "Yaw  : " + str(yawData), (20, 70), font, fontScale, yawColor, thickness, cv2.LINE_AA)
        cv2.putText(img, "Bug  : " + str(bugData), (20, 90), font, fontScale, bugColor, thickness, cv2.LINE_AA)
        cv2.putText(img, "Age  : " + str(ageData), (20, 110), font, fontScale, ageColor, thickness, cv2.LINE_AA)

        cv2.imshow('Facial Landmarks of ' + im_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imshow('5-Point Landmarks of ' + im_name, cv2.cvtColor(img5, cv2.COLOR_RGB2BGR))

        posx = 10
        posy = 10
        cv2.moveWindow('Facial Landmarks of ' + im_name, posx, posy)
        cv2.moveWindow('5-Point Landmarks of ' + im_name, posx+515, posy)

    # Saving 5 point landmarks
    if landmarks5point:
        landmarks_dir = os.path.join(photo_path + '_landmarks5p')
        landmarks_file = im_name.replace("png", "txt").replace("jpg", "txt")
        if not os.path.exists(landmarks_dir):
            os.makedirs(landmarks_dir)
        if silent_auto:
            landmarks = data['faceLandmarks']
            # Print 5-point facial landmarks
            left_eye = mean(landmarks['left_eye'])
            right_eye = mean(landmarks['right_eye'])
            nose = point(landmarks['nose'][3]['x'], landmarks['nose'][3]['y'])
            left_mouth = point(landmarks['mouth'][0]['x'], landmarks['mouth'][0]['y'])
            right_mouth = point(landmarks['mouth'][6]['x'], landmarks['mouth'][6]['y'])

        with open(os.path.join(landmarks_dir, landmarks_file), 'w') as f:
            f.write(str(right_eye[0]/2) + "\t" + str(right_eye[1]/2) + "\r" +
                    str(left_eye[0]/2) + "\t" + str(left_eye[1]/2) + "\r" +
                    str(nose[0]/2) + "\t" + str(nose[1]/2) + "\r" +
                    str(left_mouth[0]/2) + "\t" + str(left_mouth[1]/2) + "\r" +
                    str(right_mouth[0]/2) + "\t" + str(right_mouth[1]/2))

    # AUTO OR MANAL MODE
    if auto or silent_auto:
        if silent_auto:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(delay) & 0xFF
        if warning:
            send2trash.send2trash(im_path[i])
            send2trash.send2trash(lm_path[i])
            counter -= 1
            print("\n * '" + im_name + "'  and '.json' metadata file deleted !\n")

        # Press 'q' key to close
        if key == ord('q'):
            print("\n * Quit application ?\n")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                exit()
    else:
        # Manual Delete Mode ***********************
        key = cv2.waitKey(0) & 0xFF
        # Press 'd' key to delete files
        if key == ord('d'):
            print("\n * '" + im_name + "'  and '.json' metadata file will be deleted !\n")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                send2trash.send2trash(im_path[i])
                send2trash.send2trash(lm_path[i])
                counter -= 1

        # Press 'q' key to close
        if key == ord('q'):
            print("\n * Quit application ?\n")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                exit()

    cv2.destroyAllWindows()

print("\n *** End of folder ***\n")
