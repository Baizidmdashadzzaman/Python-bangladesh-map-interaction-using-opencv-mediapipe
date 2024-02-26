import pickle
import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector

######################################
cam_id = 0
width, height = 500,700
map_file_path = "../Data/map.p"
division_file_path = "../Data/division.p"
######################################

file_obj = open(map_file_path, 'rb')
map_points = pickle.load(file_obj)
file_obj.close()
print(f"Loaded map coordinates.")

# Load previously defined Regions of Interest (ROIs) polygons from a file
if division_file_path:
    file_obj = open(division_file_path, 'rb')
    polygons = pickle.load(file_obj)
    file_obj.close()
    print(f"Loaded {len(polygons)} division.")
else:
    polygons = []

# Open a connection to the webcam
cap = cv2.VideoCapture(cam_id)  # For Webcam
# Set the width and height of the webcam frame
cap.set(3, width)
# cap.set(4, height)
# Counter to keep track of how many polygons have been created
counter = 0
# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False,
                        maxHands=2,
                        modelComplexity=1,
                        detectionCon=0.5,
                        minTrackCon=0.5)

# division_distance_list = [
#     # Rangpur,Mymensignh,Sylhet,Rajshahi,Dhaka,Comilla,Kulna,Chittagong,Barishal
#     ["Dhaka", "Chittagong", "252.1 km"],["Chittagong","Dhaka", "252.1 km"]
# ]

division_distance_list = [
    # Rangpur,Mymensignh,Sylhet,Rajshahi,Dhaka,Comilla,Kulna,Chittagong,Barishal
    ["Rangpur", "Mymensignh", 235.7], ["Mymensignh", "Rangpur", 235.7],
    ["Rangpur", "Sylhet", 344.5], ["Sylhet", "Rangpur", 344.5],
    ["Rangpur", "Rajshahi", 217.2], ["Rajshahi", "Rangpur", 217.2],
    ["Rangpur", "Dhaka", 326.4], ["Dhaka", "Rangpur", 326.4],
    ["Rangpur", "Comilla", 461.9], ["Comilla", "Rangpur", 461.9],
    ["Rangpur", "Kulna", 490.3], ["Kulna", "Rangpur", 490.3],
    ["Rangpur", "Chittagong", 567.8], ["Chittagong", "Rangpur", 567.8],
    ["Rangpur", "Barishal", 580.6], ["Barishal", "Rangpur", 580.6],
    
    ["Mymensignh", "Sylhet", 109.2], ["Sylhet", "Mymensignh", 109.2],
    ["Mymensignh", "Rajshahi", 341.4], ["Rajshahi", "Mymensignh", 341.4],
    ["Mymensignh", "Dhaka", 257.4], ["Dhaka", "Mymensignh", 257.4],
    ["Mymensignh", "Comilla", 310.6], ["Comilla", "Mymensignh", 310.6],
    ["Mymensignh", "Kulna", 339.2], ["Kulna", "Mymensignh", 339.2],
    ["Mymensignh", "Chittagong", 399.6], ["Chittagong", "Mymensignh", 399.6],
    ["Mymensignh", "Barishal", 412.5], ["Barishal", "Mymensignh", 412.5],
    
    ["Sylhet", "Rajshahi", 445.8], ["Rajshahi", "Sylhet", 445.8],
    ["Sylhet", "Dhaka", 239.1], ["Dhaka", "Sylhet", 239.1],
    ["Sylhet", "Comilla", 177.3], ["Comilla", "Sylhet", 177.3],
    ["Sylhet", "Kulna", 295.9], ["Kulna", "Sylhet", 295.9],
    ["Sylhet", "Chittagong", 304.4], ["Chittagong", "Sylhet", 304.4],
    ["Sylhet", "Barishal", 349.5], ["Barishal", "Sylhet", 349.5],
    
    ["Rajshahi", "Dhaka", 242.2], ["Dhaka", "Rajshahi", 242.2],
    ["Rajshahi", "Comilla", 464.2], ["Comilla", "Rajshahi", 464.2],
    ["Rajshahi", "Kulna", 228.1], ["Kulna", "Rajshahi", 228.1],
    ["Rajshahi", "Chittagong", 500.6], ["Chittagong", "Rajshahi", 500.6],
    ["Rajshahi", "Barishal", 513.5], ["Barishal", "Rajshahi", 513.5],
    
    ["Dhaka", "Comilla", 97.9], ["Comilla", "Dhaka", 97.9],
    ["Dhaka", "Kulna", 225.6], ["Kulna", "Dhaka", 225.6],
    ["Dhaka", "Chittagong", 252.1], ["Chittagong", "Dhaka", 252.1],
    ["Dhaka", "Barishal", 272.2], ["Barishal", "Dhaka", 272.2],
    
    ["Comilla", "Kulna", 130.1], ["Kulna", "Comilla", 130.1],
    ["Comilla", "Chittagong", 155.1], ["Chittagong", "Comilla", 155.1],
    ["Comilla", "Barishal", 86.1], ["Barishal", "Comilla", 86.1],
    
    ["Kulna", "Chittagong", 246.1], ["Chittagong", "Kulna", 246.1],
    ["Kulna", "Barishal", 211.9], ["Barishal", "Kulna", 211.9],
    
    ["Chittagong", "Barishal", 210.2], ["Barishal", "Chittagong", 210.2],
]



def warp_image(img, points, size=[500, 700]):

    pts1 = np.float32([points[0], points[1], points[2], points[3]])
    pts2 = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (size[0], size[1]))
    return imgOutput, matrix


def warp_single_point(point, matrix):

    # Convert the point to homogeneous coordinates
    point_homogeneous = np.array([[point[0], point[1], 1]], dtype=np.float32)

    # Apply the perspective transformation to the point
    point_homogeneous_transformed = np.dot(matrix, point_homogeneous.T).T

    # Convert back to non-homogeneous coordinates
    point_warped = point_homogeneous_transformed[0, :2] / point_homogeneous_transformed[0, 2]
    point_warped = int(point_warped[0]), int(point_warped[1])

    return point_warped


def inverse_warp_image(img, imgOverlay, map_points):

    # Convert map_points to NumPy array
    map_points = np.array(map_points, dtype=np.float32)

    # Define the destination points for the overlay image
    destination_points = np.array([[0, 0], [imgOverlay.shape[1] - 1, 0], [0, imgOverlay.shape[0] - 1],
                                   [imgOverlay.shape[1] - 1, imgOverlay.shape[0] - 1]], dtype=np.float32)

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(destination_points, map_points)

    # Warp the overlay image to fit the perspective of the original image
    warped_overlay = cv2.warpPerspective(imgOverlay, M, (img.shape[1], img.shape[0]))

    # Combine the original image with the warped overlay
    result = cv2.addWeighted(img, 1, warped_overlay, 0.65, 0, warped_overlay)

    return result


def get_finger_location(img, imgWarped):

    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=False, flipType=True)
    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        indexFinger = hand1["lmList"][8][0:2]  # List of 21 landmarks for the first hand
        # cv2.circle(img,indexFinger,5,(255,0,255),cv2.FILLED)
        warped_point = warp_single_point(indexFinger, matrix)
        warped_point = int(warped_point[0]), int(warped_point[1])
        print(indexFinger, warped_point)
        cv2.circle(imgWarped, warped_point, 5, (255, 0, 0), cv2.FILLED)
        if len(hands) == 2:
            hand2 = hands[1]
            indexFinger2 = hand2["lmList"][8][0:2]  # List of 21 landmarks for the first hand
            warped_point2 = warp_single_point(indexFinger2, matrix)
            cv2.circle(imgWarped, warped_point2, 5, (255, 0, 64), cv2.FILLED)
            warped_point = [warped_point, warped_point2]

    else:
        warped_point = None

    return warped_point


def create_overlay_image(polygons, warped_point, imgOverlay):


    if isinstance(warped_point, list):
        check = []
        for warp_point in warped_point:
            for polygon, name in polygons:
                polygon_np = np.array(polygon, np.int32).reshape((-1, 1, 2))
                result = cv2.pointPolygonTest(polygon_np, warp_point, False)
                if result >= 0:
                    cv2.polylines(imgOverlay, [np.array(polygon)], isClosed=True, color=(0, 181, 26), thickness=2)
                    cv2.fillPoly(imgOverlay, [np.array(polygon)], (0, 181, 26))
                    cvzone.putTextRect(imgOverlay, name, polygon[0], scale=1, thickness=1)
                    # cvzone.putTextRect(imgOverlay, name, (0, 100), scale=8, thickness=5)
                    check.append(name)
        if len(check) == 2:
            cv2.line(imgOverlay, warped_point[0], warped_point[1], (0, 181, 26), 10)
            for division_distance in division_distance_list:
                if check[0] in division_distance and check[1] in division_distance:
                    cvzone.putTextRect(imgOverlay, division_distance[1] + " to " + division_distance[0], (0, 100), scale=2,
                                       thickness=5)
                    cvzone.putTextRect(imgOverlay, division_distance[2], (0, 200), scale=2, thickness=5)
    else:
        # loop through all the countries
        for polygon, name in polygons:
            polygon_np = np.array(polygon, np.int32).reshape((-1, 1, 2))
            result = cv2.pointPolygonTest(polygon_np, warped_point, False)
            if result >= 0:
                cv2.polylines(imgOverlay, [np.array(polygon)], isClosed=True, color=(0, 181, 26), thickness=2)
                cv2.fillPoly(imgOverlay, [np.array(polygon)], (0, 181, 26))
                cvzone.putTextRect(imgOverlay, name, polygon[0], scale=1, thickness=1)
                cvzone.putTextRect(imgOverlay, name, (0, 100), scale=2, thickness=5)

    return imgOverlay


while True:
    # Read a frame from the webcam
    success, img = cap.read()
    imgWarped, matrix = warp_image(img, map_points)
    imgOutput = img.copy()

    # Find the hand and its landmarks
    warped_point = get_finger_location(img, imgWarped)

    h, w, _ = imgWarped.shape
    imgOverlay = np.zeros((h, w, 3), dtype=np.uint8)

    if warped_point:
        imgOverlay = create_overlay_image(polygons, warped_point, imgOverlay)
        imgOutput = inverse_warp_image(img, imgOverlay, map_points)

    # imgStacked = cvzone.stackImages([img, imgWarped,imgOutput,imgOverlay], 2, 0.3)
    # cv2.imshow("Stacked Image", imgStacked)

    # cv2.imshow("Original Image", img)
    # cv2.imshow("Warped Image", imgWarped)
    cv2.imshow("Output Image", imgOutput)

    key = cv2.waitKey(1)