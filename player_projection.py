import cv2
import numpy as np

def compute_camera_matrix(x, y, z, u, v):
    # Fulfill a matrix with the equations that determine the camera matrix
    matrix = []
    for i in range(len(x)):
        matrix.append([x[i], y[i], z[i], 1, 0, 0, 0, 0, -u[i] * x[i], -u[i] * y[i],-u[i] * z[i], -u[i]])
        matrix.append([0, 0, 0, 0, x[i], y[i], z[i], 1, -v[i] * x[i], -v[i] * y[i],-v[i] * z[i], -v[i]])

    matrix = np.array(matrix)

    #We minimize || Am || = 0 subject to ||m|| = 1 using svd method
    u, s, vh = np.linalg.svd(matrix)

    #The values from the last row of Vh are our (matrix) values
    # so we reconstruct our homography matrix from these values
    m = np.array([
        [vh[-1][0], vh[-1][1], vh[-1][2], vh[-1][3]],
        [vh[-1][4], vh[-1][5], vh[-1][6], vh[-1][7]],
        [vh[-1][8], vh[-1][9], vh[-1][10], vh[-1][11]],
        ])

    return m

def calculate_projection_matrix():
    # We define our calibration points in world and image coordinates
    # Calibration points
    image_points = np.array([
    [275,84],      # escanteio superiors
    [227,107],     # borda exterior superior area pequena
    [159,141],     # trave superior ou esquerda
    [125,158],     # trave inferior ou direita
    [158,112],     # angulo superior ou da esquerda da goleira
    [124,128],     # angulo inferior ou direito da goleira
    [31,205],      # Canto inferior direito da grande area
    [250,222],     # Canto inferior direito da pequena area
    [241,132],     # borda externa superior ou esquerda da pequena area
    [160,177]      # borda externa inferior ou direita da pequena area
    ])

    world_points = np.array([
    [0, 0, 0],     # escanteio superiors
    [13.84, 0, 0],     # borda exterior superior area pequena
    [30.34, 0, 0],     # trave superior ou esquerda
    [37.66, 0, 0],     # trave inferior ou direita
    [30.34, 0, 2.44],  # angulo superior ou da esquerda da goleira
    [37.66, 0, 2.44],  # angulo inferior ou direito da goleira
    [54.16, 0, 0],     # Canto inferior direito da grande area
    [54.16, 16.5, 0],  # Canto inferior direito da pequena area
    [24.84, 5.5, 0],   # borda externa superior ou esquerda da pequena area
    [43.16, 5.5, 0]    # borda externa inferior ou direita da pequena area
    ])

    x = world_points[:, 0]
    y = world_points[:, 1]
    z = world_points[:, 2]
    u = image_points[:, 0]
    v = image_points[:, 1]

    return compute_camera_matrix(x, y, z, u, v)

#We transform our world point/coordinate to a pixel coordinate using the dot producit with the projection matrix
def transform_world_point(point, matrix):
    #We convert the point to homogeneous coordinates adding one more dimension with value 1 to our vector
    homogeneous_coordinates = np.array([point[0], point[1], point[2], 1])
    dot_product = np.dot(matrix, homogeneous_coordinates)
    s = dot_product[-1] # s = last element of the array
    coordinates = dot_product / s # --> (x/s, y/s 1)
    pixel_point = coordinates[0], coordinates[1]
    return pixel_point

#We transform our pixel coordinate to a world point/coordinate using the dot producit with the adapted inverse matrix
def transform_pixel_point(point, inverse_matrix):
    #We convert the point to homogeneous coordinates adding one more dimension with value 1 to our vector
    homogeneous_coordinates = np.array([point[0], point[1], 1])
    dot_product = np.dot(inverse_matrix, homogeneous_coordinates)
    s = dot_product[-1] # s = last element of the array
    coordinates = dot_product / s # --> (x/s, y/s, 1)
    world_point = coordinates[0], coordinates[1], 0 # we set z = 0 to complete our new 3d coordinates
    return world_point

def calculate_inverse_matrix(matrix):
    #We want to transform a pixel coordinate (2d) in world coordinate (3d)
    #Our matrix has shape 3,4 so we have to handle it deleting
    #the column corresponding to z to be able to find the inverse matrix
    #and also beucase our image point has only 2 dimension, 3 in homogeneous coordinates
    # for when we want to calculate the dot product
    aux_matrix = np.copy(matrix)
    aux_matrix = np.delete(aux_matrix, 2, 1)
    inverse_matrix = np.linalg.inv(aux_matrix)
    return inverse_matrix

def calculate_player_points(matrix, x, y):
    #Calculates inverse matrix to transform from pixel to world coordinates
    inverse_matrix = calculate_inverse_matrix(matrix)

    #Tansforms the selected point from pixel to world coordinates
    world_point = transform_pixel_point((x, y), inverse_matrix)

    # To draw the player line we will have to find an additional point, "his head"
    # So we simply keep x and y constant and use z = 1.8 to represent where it would be
    head_point = world_point[0], world_point[1], 1.8

    # Now we transform the head point from world coordinates to pixel coordinates
    # The other point we already have it because it`s the one we received
    head_pixel_point = transform_world_point(head_point, matrix)

    return int(round(head_pixel_point[0])), int(round(head_pixel_point[1]))


def mouse_drawing(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        editingImage = data["image"].copy()
        x2,y2 = calculate_player_points(data["matrix"], x, y) #We only need to find his head coordinate
        cv2.line(editingImage,(x, y), (x2, y2), (0, 0, 255), 2) # The other coordinate is the x,y of the mouse click event
        cv2.imshow("Maracana", editingImage)


# função main em python
if __name__ == '__main__' :
    #We define a dic with our projection matrix and image
    data = {}
    data["matrix"] = calculate_projection_matrix()
    data["image"] = cv2.imread('maracana1.jpg') # Carrega e mostra a imagem

    # We show the image and set the callback to draw the line when clicked on the image
    cv2.imshow("Maracana", data["image"])
    cv2.setMouseCallback("Maracana", mouse_drawing, data)
    key = cv2.waitKey(0)
