import numpy as np
import cv2
import cv2.aruco as aruco



def single_marker_estimation():
    # setting up the video feed source, dictionary of aruco markers and dictionary parameters
    cap = cv2.VideoCapture(0)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    # intrinsic camera specific parameter matrix and distortion coefficients for correct pose estimation
    mtx = np.array([[5.3434144579284975e+02, 0., 3.3915527836173959e+02],
                    [0., 5.3468425881789324e+02, 2.3384359492532246e+02],
                    [0., 0., 1.]], float)
    dist = np.array([-2.8832098285875657e-01, 5.4107968489116441e-02,
                     1.7350162244695508e-03, -2.6133389531953340e-04,
                     2.0411046472667685e-01], float)

    while True:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # library use to detect aruco markers in a given frame with dictionary parameters
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is not None:  # if aruco marker detected
            # single aruco marker pose estimation
            rvec, tvec, objp = aruco.estimatePoseSingleMarkers(corners, 1, mtx, dist)

            # draw the detected markers and x,y,z axis on the frame
            #aruco.drawDetectedMarkers(frame, corners, ids)
            #aruco.drawAxis(frame, mtx, dist, rvec, tvec, 1)

            # create points for a three-dimensional cube
            axis = np.float32([[0, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, 0, 0],
                               [0, 0, 1], [0, -1, 1], [-1, -1, 1], [-1, 0, 1]])

            # projects three-dimensional points to a two-dimensional plane with given rotation and tranformation
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

            # reshapes the array into a matrix with 2 columns with disregarding the number of rows
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # draw the lines from given points
            frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 3)
            for i, j in zip(range(4), range(4, 8)):
                frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
            frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 3)

        # display current frame in a window
        cv2.imshow('frame', frame)

        # exit loop on key press 'q' and exit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    single_marker_estimation()
