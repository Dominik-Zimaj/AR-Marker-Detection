import cv2
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from threading import Thread

texture_id = 0
thread_quit = 0
X_AXIS = 0.0
Y_AXIS = 0.0
Z_AXIS = 0.0
DIRECTION = 1
cap = cv2.VideoCapture(0)
new_frame = cap.read()[1]


def init():
    video_thread = Thread(target=update, args=())
    video_thread.start()


def init_gl(width, height):
    global texture_id
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_TEXTURE_2D)
    texture_id = glGenTextures(1)


def update():
    global new_frame
    while (True):
        new_frame = cap.read()[1]
        if thread_quit == 1:
            break
    cap.release()
    cv2.destroyAllWindows()


def draw_gl_scene():
    global cap
    global new_frame
    global X_AXIS, Y_AXIS, Z_AXIS
    global DIRECTION
    global texture_id
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    frame = new_frame
    # convert image to OpenGL texture format
    tx_image = cv2.flip(frame, 0)
    tx_image = Image.fromarray(tx_image)
    ix = tx_image.size[0]
    iy = tx_image.size[1]
    tx_image = tx_image.tobytes('raw', 'BGRX', 0, -1)
    # create texture
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tx_image)

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glPushMatrix()
    glTranslatef(0.0, 0.0, -6.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-4.0, -3.0, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(4.0, -3.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(4.0, 3.0, 0.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-4.0, 3.0, 0.0)
    glEnd()
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0.0, 0.0, -6.0)
    glRotatef(X_AXIS, 1.0, 0.0, 0.0)
    glRotatef(Y_AXIS, 0.0, 1.0, 0.0)
    glRotatef(Z_AXIS, 0.0, 0.0, 1.0)

    glBegin(GL_QUADS)

    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(1.0, -1.0, -1.0)

    glColor3f(0.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, -1.0)

    glColor3f(1.0, 1.0, 1.0)
    glEnd()
    glPopMatrix()
    X_AXIS = X_AXIS - 0.30
    Z_AXIS = Z_AXIS - 0.30

    glutSwapBuffers()


def key_pressed(key, x, y):
    global thread_quit
    if key == chr(27) or key == "q":
        thread_quit = 1
        sys.exit()


def run():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutInitWindowPosition(200, 200)
    window = glutCreateWindow('My and Cube')
    glutDisplayFunc(draw_gl_scene)
    glutIdleFunc(draw_gl_scene)
    glutKeyboardFunc(key_pressed)
    init_gl(640, 480)
    glutMainLoop()


if __name__ == "__main__":
    init()
    run()
