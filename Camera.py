import cv2 as cv

class Camera:

    #Constructor for camera
    def __init__(self):
        self.camera = cv.VideoCapture(0) #Currently only one camera hence using 0
        if not self.camera.isOpened():
            raise ValueError("Unable to open the camera")

        self.width = self.camera.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv.CAP_PROP_FRAME_HEIGHT)

    #Destructor for camera
    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    #Function to get a next frame
    def get_Frame(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read() #returns the current frame

            if ret:
                return(ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            else:
                return(ret,None)
        else:
            return None