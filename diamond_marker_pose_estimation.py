import numpy as np
import cv2
import cv2.aruco as aruco


def diamond_marker_estimation():
    cap = cv2.VideoCapture(0)
    # parameters that are in relation to the marker sizes compared to the chess squares
    square_length = 0.40
    marker_length = 0.25
    # intrinsic camera specific parameter matrix and distortion coefficients for correct pose estimation
    mtx = np.array([[5.3434144579284975e+02, 0., 3.3915527836173959e+02],
                    [0., 5.3468425881789324e+02, 2.3384359492532246e+02],
                    [0., 0., 1.]], float)
    dist = np.array([-2.8832098285875657e-01, 5.4107968489116441e-02,
                     1.7350162244695508e-03, -2.6133389531953340e-04,
                     2.0411046472667685e-01], float)

    while True:
        ret, frame = cap.read()
        # create a dictionary and an charuco diamond image
        diamond_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        diamond_frame = aruco.drawCharucoDiamond(diamond_dict, np.array([1, 1, 1, 1]), 200, 120)
        # create standard parameters and detect present markers in a frame with a specific dictionary
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = aruco.detectMarkers(frame, diamond_dict, parameters=parameters)

        if ids is not None:
            # detect a charuco diamond from detected single markers and draw the diamond
            diamond_corners, diamond_ids = aruco.detectCharucoDiamond(frame, corners, ids,
                                                                      square_length / marker_length)
            aruco.drawDetectedDiamonds(frame, diamond_corners, diamond_ids)
            # estimate the pose of every marker and for every calculated rotation vector
            rvecs, tvecs, obj_points = aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)
            for i in range(len(rvecs)):
                cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 0.3)
        # show 2 frames for the generated charuco diamond and the webcam feed with pose estimation
        cv2.imshow("diamond", diamond_frame)
        cv2.imshow("frame", frame)

        # exit loop on key press 'q' and exit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    diamond_marker_estimation()
