import cv2
import numpy as np
import table_classes


def cv_identify_line_contours(image_path: str, iterations: int = 5) -> tuple:
    """
    Get table contours from image. Process provided image with openCV package (cv2).
    Identify horizontal and vertical lines in image, extract contours and identify
    position of table cells.

    :param image_path: string with path to file
    :param iterations: number of iterations for erosion and dilation
    :return: list with identified table contours, each contour is array with [x, y]
             coordinates of rectangle corners.
    """
    img = cv2.imread(image_path)
    # before thresholding transform image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # perform thresholding, by selected threshold value (now 128, it is right
    # in the middle of 8-bit value => 0-255) change value of pixel either to 1
    # or 0. After this step we have just black and white picture
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    _, img_bin_result = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    # invert picture => black to white and white to black
    img_bin = 255 - img_bin

    # if number of iterations wasn't provided we can use our default 0.15% of image pixel height. For 300 dpi it means
    # that we will have 5 iterations, that should work well.
    if not iterations:
        h, _ = img.shape
        iterations = round(h * 0.0015)

    # kernel is matrix by multiplying bits to change original image in some
    # desired way. For example, it can be used to dilate or threshold image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1] // 150))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1] // 150, 1))

    vertically_eroded_img = cv2.erode(img_bin, vertical_kernel, iterations=iterations)
    vertical_lines = cv2.dilate(vertically_eroded_img, vertical_kernel, iterations=iterations)

    horizontally_eroded_img = cv2.erode(img_bin, horizontal_kernel, iterations=iterations)
    horizontal_lines = cv2.dilate(horizontally_eroded_img, horizontal_kernel, iterations=iterations)

    # cv2.addWeighted() produce weighted sum of two pictures, this time give both
    # same weight
    vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    # ~ tilde should mean bitwise "not" operator, it flips black and white in picture
    vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=iterations)

    _, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,
                                                 128, 255,
                                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # find contours in picture
    # mode parameter cv2.RETR_TREE returns contours in parent-child structure
    # method parameter cv2.CHAIN_APPROX_SIMPLE is not storing all points of contour
    # but only endpoints of contour shape (for example corners of rectangle)
    contours, _ = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, img_bin_result


def filter_duplicates_from_contours(contours: tuple, tolerance: int) -> list:
    """
    Contours are result of image processing. Lines in picture were identified and contours
    return at this point of process. Each contour contain multiple points, typically there
    are 4 points if shape is just rectangle. Each point is corner. There could be more points
    which is telling us that structure is more complicated than just rectangle, it could
    indicate presence of table. This function drops duplicitous points, points which are
    closer to each other than tolerance parameter, keep just distinct points. To simplify
    next step of processing each point is given row index to be able to sort point in each
    contour.

    :param contours: tuple with contours.
    :param tolerance: threshold value, how close points in contour could be to consider it as duplicates. Contour
           distance is measure as number of pixels between them.
    :return: duplicate free list with contours, points in each contour are sorted according to x coord and row.
    """
    # contours are result of _cv_identify_line_contours() function, it should be passed
    # as argument of this function. Format of contours is (np.array([[[x1, y1]], [[x2, y2]], [[x3, y3]], ...])))
    # typically there should be at least 4 points, could be even more than that
    sorted_contours = []
    for contour in contours:
        # in this for loop drop all points which are too close to each other
        list_contour = list(contour)
        distinct_contour = []
        for index, point in enumerate(list_contour):
            duplicates = []
            # check whether one of subsequent points in contour isn't closer than tolerance
            for index_next_point in range(index + 1, len(list_contour)):
                if abs(point[0][0] - list_contour[index_next_point][0][0]) < tolerance and \
                        abs(point[0][1] - list_contour[index_next_point][0][1]) < tolerance:
                    duplicates.append(index_next_point)
            # Now we can identify as duplicate only next elements in list_contour
            # than iterator is pointing at currently. So if we delete them iterator
            # won't loop through them, but this is what was intended
            for index_contour in sorted(duplicates, reverse=True):
                del list_contour[index_contour]

            # all other duplicitous points were dropped, so we can append
            distinct_contour.append(point[0].tolist())

        # it would be easier to work with points in contour list if we could somehow
        # sort it. Good approach is to create horizontal structure with giving row
        # index to each point. Then we could sort it by row and x coord. So in next
        # step we can process it row by row and identify boxes out of points in
        # contours
        row = 0
        value = -10 ** 5
        to_sort_contour = []
        for point in sorted(distinct_contour, key=lambda x: round(list(x)[1] / tolerance)):
            if point[1] - value >= tolerance:
                row += 1
                value = point[1]
            new_point = (point[0], point[1], row)
            to_sort_contour.append(new_point)

        # sort contours by row and x coordinate
        sorted_contours.append(sorted(to_sort_contour, key=lambda x: (x[2], x[0])))

    return sorted_contours


def create_cells(sorted_contours: list, tolerance: int) -> list:
    """
    Create list with cells in format [x, y, width, height], where x, y are coordinates
    of left top corner of box (rectangle)

    :param sorted_contours: result of function filter_duplicates_from_contours, with
    list of contours where points in each contour are sorted by row and x coord
    :param tolerance: threshold value, how close points in contour could be to consider it as duplicates
    :return: list with boxes separated into lists by contours
    """
    cells_by_contour = []
    for index_contour, contour in enumerate(sorted_contours):
        cells_by_contour.append([])
        for index_point, point in enumerate(contour):
            # it doesn't make sense to create rectangles with less than 4 corners
            # TODO: actually there might be tables without outline border lines and
            #  with this approach we could loose corner cells of table, investigate this!
            if index_point + 1 > len(contour) - 3:
                break
            right_point = contour[index_point + 1]
            # this condition is enough after sorting and filtering duplicates it is not ambiguous
            if right_point[1] - point[1] < tolerance:
                # now we know that there is another contour point in same row on the right side of point
                # now we need to find point bellow
                bellow_point = None
                for index_next_point in range(index_point + 2, len(contour)):
                    # because points are sorted and "duplicate" this is enough
                    if contour[index_next_point][0] - point[0] < tolerance:
                        bellow_point = contour[index_next_point]
                        break

                if bellow_point and right_point[0] - point[0] > 0 and bellow_point[1] - point[1] > 0:
                    cell = table_classes.Cell(point[0], point[1], right_point[0] - point[0], bellow_point[1] - point[1])
                    cells_by_contour[index_contour].append(cell)

    return cells_by_contour


def sort_cells(cells_by_contour: list, tolerance: int, img) -> list:
    """
    Sort boxes by its sizes and delete a largest box if it is whole page

    :param cells_by_contour: list with boxes divided by contours
    :param tolerance: threshold value, how close points in contour could be to consider it as duplicates
    :param img: image from contours
    :return: list with boxes sorted by size
    """
    # place all boxes into one list
    cells = []
    for contour in cells_by_contour:
        for cell in contour:
            cells.append(cell)
    # sort boxes from larges to smallest
    sorted_cells = sorted(cells, key=lambda x: x.width * x.height, reverse=True)

    # first (largest) box is most likely whole page, we aren't interested in that box
    # we can check it and delete it
    h, w = img.shape
    largest_cell = sorted_cells[0]
    # if largest_cell[0] < tolerance and largest_cell[1] < tolerance and \
    if abs(largest_cell.x + largest_cell.width - w) < tolerance and \
            abs(largest_cell.y + largest_cell.height - h) < tolerance:
        del sorted_cells[0]

    return sorted_cells


def nested_structure(sorted_cells: list) -> dict:
    """
    Find nested structure of cells. Some cells can contain other cells inside (it might be more tables in a single
    page), we could call this cell super set of contained cell (sub sets). Find out which cells are super sets help
    us with learning about table structure.

    :param sorted_cells: list with cells sorted by size.
    :return: dict with all cells as key and lists with sub set boxes as values.
    """
    # check if some rectangles contain another rectangles from list
    super_sets = {}
    for cell_index, cell in enumerate(sorted_cells):
        super_sets[cell_index] = []
        for index_cell_to_compare in range(cell_index + 1, len(sorted_cells)):

            cell_to_compare = sorted_cells[index_cell_to_compare]
            if cell.x + cell.width >= cell_to_compare.x >= cell.x and \
                    cell.x + cell.width >= cell_to_compare.x + cell_to_compare.width >= cell.x and \
                    cell.y + cell.height >= cell_to_compare.y >= cell.y and \
                    cell.y + cell.height >= cell_to_compare.y + cell_to_compare.height >= cell.y:
                # append cell as subset cell (index_cell_to_compare) of current superset (index_cell)
                super_sets[cell_index].append(index_cell_to_compare)

    # check whether all sub-boxes in super-box are "neighbours" and whether
    # all together fill whole super-box (minus tolerance), otherwise we would
    # handle some boxes twice
    for index in super_sets.keys():
        for index_sub_set, sub_box in enumerate(super_sets[index]):
            if super_sets[sub_box]:
                for item in super_sets[sub_box]:
                    try:
                        while True:
                            super_sets[index].remove(item)
                    except ValueError:
                        pass

    return super_sets


def locate_table(filepath: str, tolerance_perc: float, iterations: int = None) -> tuple:
    """
    Identify table inside of image, then return supersets and sorted list of cells

    :param filepath: string with path to file
    :param tolerance_perc: threshold value in percentage of image pixel height, how close points in contour could be
                           considered as overlapping
    :param iterations: number of iterations used for of dilation and erosion of image to extract table contours
    :return: sorted list with cells from all contours and dict with superset cells
    """
    contour, img = cv_identify_line_contours(filepath, iterations)

    h, _ = img.shape
    tolerance = round(h * tolerance_perc)

    contour_list = filter_duplicates_from_contours(contour, tolerance)
    cell_list = create_cells(contour_list, tolerance)
    sorted_cells = sort_cells(cell_list, tolerance, img)
    super_sets = nested_structure(sorted_cells)

    return sorted_cells, super_sets, img


def merge_cells_in_sequence(cells_list: list, tolerance: int) -> list:
    """
    Get merged cells from whole sequence of boxes

    :param cells_list: list with cells, each cell has 4 attributes - x, y, width and height
    :param tolerance: int value representing maximal allowed distance between cells to be still considered as
           neighbours.
    :return:
    """
    # Merge all boxes together, starting with first box in the list
    merged_box = cells_list[0]
    for box in cells_list:
        # little side effect in this for loop is that we are merging in first
        # step very first box with itself, but basically it will do nothing
        merged_box = merged_box.merge_2_boxes(box, tolerance)

    return merged_box


def merge_neighbour_cells(cells_list: list, tolerance: int) -> list:
    """
    Given list of boxes merge all possible neighbour boxes together and
    return new list with merged boxes

    :param cells_list: list with cells, each cell has 4 attributes - x, y, width and height
    :param tolerance: int value representing maximal allowed distance between boxes to be still considered as
                      neighbours.
    :return: list with merged boxes
    """
    # we will merge boxes row by row
    # first add row indicator to each box
    row = 0
    value = -1  # it's safe to use start value -1 because y coordinate cannot be negative
    cell_tup_list = []
    for cell in sorted(cells_list, key=lambda _cell: (_cell.y, _cell.x)):
        # Check y coordinate, if it exceeded threshold (tolerance) value we know it belong to the new row.
        # We can append it to list with added row information.
        if cell.y - value >= tolerance:
            row += 1
            value = cell.y
        cell_tuple = (cell, row)
        cell_tup_list.append(cell_tuple)

    sorted_cell_list = sorted(cell_tup_list, key=lambda _cell_tuple: (_cell_tuple[1], _cell_tuple[0].x))

    # loop through all rows, we start with row number equal to 1 up to value currently stored in row variable
    merged_row_list = []
    for row_ind in range(1, row + 1):
        row_list = []
        for cell_tuple in sorted_cell_list:
            if cell_tuple[1] == row_ind:
                row_list.append(cell_tuple[0])
        merged_row_list.append(merge_cells_in_sequence(row_list.copy(), tolerance))

    return merge_cells_in_sequence(merged_row_list, tolerance)


def supersets_differences(cells_list: list, super_sets: dict, tolerance: int) -> dict:
    """
    Get difference between each super set box and it's subset boxes

    :param cells_list: list with cells, each cell has 4 attributes - x, y, width and height
    :param super_sets: dict, with each box as key and it's subset boxes as values
    :param tolerance: int value representing maximal allowed distance
    between boxes to be still considered as neighbours.
    :return: dict with difference boxes
    """
    differences = {}
    for super_set in super_sets:
        superset_cell = cells_list[super_set]
        # first merge subset boxes for each superset
        subset_cells = []
        for index in super_sets[super_set]:
            subset_cells.append(cells_list[index])
        if subset_cells:
            merged_subset_cell = merge_neighbour_cells(subset_cells, tolerance)
        else:
            differences[super_set] = [superset_cell]
            continue
        calculated_difference = superset_cell.cell_difference(merged_subset_cell, tolerance)
        if any(calculated_difference.values()):
            differences[super_set] = [cell for cell in calculated_difference.values() if cell]

    return differences


def find_rows_columns(cells_list: list, tolerance: int) -> list:
    """
    Add row and column information to each cell in list

    :param cells_list: list with cells, each cell has 4 attributes - x, y, width and height
    :param tolerance: int value representing maximal allowed distance between boxes to be still considered as
                      neighbours.
    :return: list with tuples, each tuple represent one of original cells. It has structure (cell, (row, column))
    """
    # adding row index into cell_tuple
    row = 0
    value = -1  # it's safe to use start value -1 because y coordinate cannot be negative
    cell_tup_list = []
    for cell in sorted(cells_list, key=lambda _cell: (_cell.y, _cell.x)):
        # Check y coordinate, if it exceeded threshold (tolerance) value we know it belong to the new row.
        # We can append it to list with added row information.
        if cell.y - value >= tolerance:
            row += 1
            value = cell.y
        cell_tuple = (cell, row)
        cell_tup_list.append(cell_tuple)

    # adding column index into cell_tuple
    column = 0
    value = -1  # it's safe to use start value -1 because x coordinate cannot be negative
    result_cell_tup_list = []
    for cell_tuple in sorted(cell_tup_list, key=lambda _cell_tuple: (_cell_tuple[0].x, _cell_tuple[1])):
        # Check y coordinate, if it exceeded threshold (tolerance) value we know it belong to the new column.
        # We can append it to list with added column information.
        if cell_tuple[0].x - value >= tolerance:
            column += 1
            value = cell_tuple[0].x
        new_cell_tuple = (cell_tuple[0], (cell_tuple[1], column))
        result_cell_tup_list.append(new_cell_tuple)

    return result_cell_tup_list
