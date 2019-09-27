import cv2
import numpy as np


def homography_2d(x, y, u, v):
    h = np.array([
        [x[0], y[0], 1, 0, 0, 0, -u[0] * x[0], - u[0] * y[0]],
        [0, 0, 0, x[0], y[0], 1, -v[0] * x[0], - v[0] * y[0]],
        [x[1], y[1], 1, 0, 0, 0, -u[1] * x[1], - u[1] * y[1]],
        [0, 0, 0, x[1], y[1], 1, -v[1] * x[1], - v[1] * y[1]],
        [x[2], y[2], 1, 0, 0, 0, -u[2] * x[2], - u[2] * y[2]],
        [0, 0, 0, x[2], y[2], 1, -v[2] * x[2], - v[2] * y[2]],
        [x[3], y[3], 1, 0, 0, 0, -u[3] * x[3], - u[3] * y[3]],
        [0, 0, 0, x[3], y[3], 1, -v[3] * x[3], - v[3] * y[3]],
    ])
    b = np.array([u[0], v[0], u[1], v[1], u[2], v[2], u[3], v[3]])

    m = np.linalg.solve(h, b)
    p = np.array([[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], 1]])
    return p


def calculate_transformation_matrix():

    pts_src = np.array([[269, 23], [264, 344], [439, 23], [585, 346]])
    pts_dst = np.array([[0, 0], [0, 68], [11, 0], [11, 68]])

    x = pts_dst[:, 0]
    y = pts_dst[:, 1]
    u = pts_src[:, 0]
    v = pts_src[:, 1]

    return homography_2d(x, y, u, v)


def calculateOffsidePoints(matrix,x,y):
    inverseMatrix = np.linalg.inv(matrix)
    homogeneousCoordinates = np.array([x,y,1])
    dotProduct = np.dot(inverseMatrix,homogeneousCoordinates)
    s = dotProduct[-1]
    coordinates = dotProduct / s # (x/s, y/s, 1)
    x = coordinates[0]

    point1 = np.array([x,0,1])
    point2 = np.array([x,68,1])

    dotProduct = np.dot(matrix,point1)
    s = dotProduct[-1]
    coordinates = dotProduct / s # (x/s, y/s, 1)
    coordinate1 = coordinates[:-1]

    dotProduct = np.dot(matrix,point2)
    s = dotProduct[-1]
    coordinates = dotProduct / s # (x/s, y/s, 1)
    coordinate2 = coordinates[:-1]

    coordinates = []
    coordinates.append(np.round(coordinate1,0).astype(int))
    coordinates.append(np.round(coordinate2,0).astype(int))

    return coordinates[0][0],coordinates[0][1],coordinates[1][0],coordinates[1][1]


def mouse_drawing(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        editedImage = params["image"].copy()
        x1,y1,x2,y2 = calculateOffsidePoints(params["matrix"],x,y)
        print(x1,y1,x2,y2)
        cv2.line(editedImage,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("Maracana", editedImage)



# função main em python
if __name__ == '__main__' :

    params = {}
    params["matrix"] = calculate_transformation_matrix()
    params["image"] = cv2.imread('maracana2.jpg') # Carrega e mostra a imagem


    cv2.imshow("Maracana", params["image"])
    cv2.setMouseCallback("Maracana", mouse_drawing, params)# Seta callback para quando for apertado com mouse
    key = cv2.waitKey(0) # Fecha janela grafica quando aperta ESC
