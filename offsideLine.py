import cv2
import numpy as np

HEIGHT = 75

# Camera matrix to relate image (2D) points with world (2D) points
# matrix =
#     [x[0], y[0], 1, 0, 0, 0, -u[0] * x[0], - u[0] * y[0], -u[0]],
#     [0, 0, 0, x[0], y[0], 1, -v[0] * x[0], - v[0] * y[0], -v[0]],
#     [x[1], y[1], 1, 0, 0, 0, -u[1] * x[1], - u[1] * y[1], -u[1]],
#     [0, 0, 0, x[1], y[1], 1, -v[1] * x[1], - v[1] * y[1], -v[1]]...
#
def compute_homography_2d(u, v, x, y):
    # Fulfill a matrix with the equations that determine the camera matrix
    matrix = []
    for i in range(len(x)):
        matrix.append([x[i], y[i], 1, 0, 0, 0, -u[i] * x[i], - u[i] * y[i], -u[i]])
        matrix.append([0, 0, 0, x[i], y[i], 1, -v[i] * x[i], - v[i] * y[i], -v[i]])

    #We minimize || Am || subject to ||m|| = 1 using svd method
    matrix = np.array(matrix)
    u, s, vh = np.linalg.svd(matrix)
    last_line_vh = vh[-1]

    #The 8 values from the last row of Vh are our answer values
    # so we reconstruct our homography matrix from these values
    h = np.array([
        [vh[8][0], vh[8][1], vh[8][2]],
        [vh[8][3], vh[8][4], vh[8][5]],
        [vh[8][6], vh[8][7], vh[8][8]],
        ])

    return h

def calculate_transformation_matrix():
    # We define our calibration points in world and image coordinates
    world_points = np.array([[0, 0], [0, HEIGHT], [11, 0], [11, HEIGHT]])
    image_points = np.array([[269, 25], [266, 346], [440, 27], [586, 346]])

    x = world_points[:, 0]
    y = world_points[:, 1]
    u = image_points[:, 0]
    v = image_points[:, 1]

    return compute_homography_2d(u, v, x, y)

def transformate_point(point, transformation_matrix):
    #We transform the point to homogeneous coordinates adding one more dimension with value 1 to our vector
    homogeneous_coordinates = np.array([point[0], point[1], 1])
    dot_product = np.dot(transformation_matrix, homogeneous_coordinates)
    s = dot_product[-1] # S = last element of the array
    coordinates = dot_product / s # (x/s, y/s, 1)
    x,y = coordinates[0], coordinates[1]
    return (x,y)

def calculate_offside_points(matrix,x,y):
    #Calculates inverse matrix to transform from image to world coordinates
    inverse_matrix = np.linalg.inv(matrix)

    #Tansforms the selected point from image to world coordinates
    world_point = transformate_point((x,y), inverse_matrix)
    x = world_point[0]

    #To draw the offside line we will have to find 2 points
    #for that we simply mantain const our x of the world coordinate and use the minimum and
    #maximum possible value for y. So we have a straight line crossing our selected point
    point1 = np.array([x, 0])
    point2 = np.array([x, HEIGHT])

    # Now we transform each of that 2 points to image coordinates
    offside_point_1 = transformate_point(point1, matrix)
    offside_point_2 = transformate_point(point2, matrix)

    return int(round(offside_point_1[0])), int(round(offside_point_1[1])), int(round(offside_point_2[0])), int(round(offside_point_2[1]))


def mouse_drawing(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        editedImage = data["image"].copy()
        x1,y1,x2,y2 = calculate_offside_points(data["matrix"], x, y)
        print(x1,y1,x2,y2)
        cv2.line(editedImage,(x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Maracana", editedImage)



# função main em python
if __name__ == '__main__' :
    #We define a dic with our transformation matrix and image
    data = {}
    data["matrix"] = calculate_transformation_matrix()
    data["image"] = cv2.imread('maracana2.jpg') # Carrega e mostra a imagem

    # We show the image and set the callback to draw the line when clicked on the image
    cv2.imshow("Maracana", data["image"])
    cv2.setMouseCallback("Maracana", mouse_drawing, data)
    key = cv2.waitKey(0)
