#Building the homography matrix
import numpy
#takes in the original board and the rotated board
def create_h_matrix(originalBoard3D, rotatedBoard):
    #3D to 2D board to get x, y, z coordinates
    originalBoard = hom_3Dto2D(originalBoard3D)
    h_matrix = np.array()
    #original
    for c in range(0, len(originalBoard[0])):
        #original matrix
        x = originalBoard[0,c]
        y = originalBoard[1,c]
        w = originalBoard[2,c]
        #rotated matrix
        _x = rotatedBoard[0,c]
        _y = rotatedBoard[1,c]
        _w = rotatedBoard[2,c]

        point_matrix = numpy.array([0, 0, 0, -_w*x, -_w*y, -_w*w, -_y*x, -_y*y, -_y*w],
                               [_w*x, _w*y, _w*w, 0, 0, 0, -_x*x, -_x*y, -x*w])
        h_matrix = np.concatenate(point_matrix);

    return h_matrix
