import table_locator
import table_ocr
import table_classes
import fitz


def expand_supersets(cell_differences: dict, sorted_cells: list, super_sets: dict, tolerance: int) -> tuple:
    """
    Add additional boxes found in box_differences list into super_sets.

    :param cell_differences: dict with cell differences
    :param sorted_cells: list with all cells, without difference cells
    :param super_sets: dict with nested superset structure
    :param tolerance: int value representing maximal allowed distance between cells to be still considered as neighbours
    :return: list with all boxes including difference boxes and superset dict including difference boxes
    """
    for difference in cell_differences:
        number_of_subset_cells = len(cell_differences[difference])
        number_of_cells = len(sorted_cells)
        if number_of_subset_cells > 1:
            cells_to_merge = []
            for cell in super_sets[difference]:
                cells_to_merge.append(sorted_cells[cell])
            additional_cell = table_locator.merge_neighbour_cells(cells_to_merge, tolerance)
            additional_cells = cell_differences[difference].copy()
            additional_cells.append(additional_cell)
            # switch original bigger cell in super_sets for of additional boxes, which are subsets of this cell
            index_range = range(number_of_cells, number_of_cells + len(additional_cells))
            for index_superset, superset in enumerate(super_sets.values()):
                if difference in superset:
                    super_sets[index_superset].remove(difference)
                    super_sets[index_superset] += list(index_range)

            sorted_cells += additional_cells

            for index in index_range:
                super_sets.setdefault(index, [])
    return sorted_cells, super_sets


def find_root_cells(super_sets: dict) -> list:
    """
    Find "root" cells, we can consider them as a separate tables

    :param super_sets: dict with supersets
    :return: list with indexes of root boxes
    """
    sub_sets = []
    for sets in super_sets.values():
        sub_sets += sets

    # root boxes are considered as separate tables in page
    root_boxes = []
    for set_index in super_sets:
        if set_index not in sub_sets:
            root_boxes.append(set_index)

    return root_boxes


def find_cell_in_dict(cell, cell_dict):
    for cell_index in cell_dict.keys():
        if cell.x == cell_dict[cell_index].x and \
           cell.y == cell_dict[cell_index].y and \
           cell.width == cell_dict[cell_index].width and \
           cell.height == cell_dict[cell_index].height:
            return cell_index
    else:
        return None


def find_table_cells(super_sets: dict, root_cells: list, sorted_cells: list, tolerance: int) -> list:
    """
    Find first level cells, this structure is considered as table structure.
    Therefore first_level should contain list with cell for each cell in table.

    :param super_sets: dict with supersets
    :param root_cells: list with root cells
    :param sorted_cells: list with all cells
    :param tolerance: int value representing maximal allowed distance between cells to be still considered as neighbours
    :return: list with indexes of cells creating table structure
    """
    # cells that are directly subsets of root cell are considered as first level
    # of table. Normally it would mean whole table structure. But page sometimes
    # contain cell which contains another table

    root_cell_list = []
    root_cell_dict = {}
    for root_index, root_set in enumerate(root_cells):
        root_cell_list.append((sorted_cells[root_set]))
        root_cell_dict[root_set] = sorted_cells[root_set]
    sorted_root_cell_list = sorted(create_root_structure(root_cells, sorted_cells, tolerance),
                                   key=lambda x: (x[1][0], x[1][1]))

    sorted_root_cell = [root_cell[0] for root_cell in sorted_root_cell_list]
    sorted_root_cells = [find_cell_in_dict(root_cell, root_cell_dict) for root_cell in sorted_root_cell]

    table_cells_index_list = []
    for root_set in sorted_root_cells:
        table_cells_index_list.append(super_sets[root_set])

    return table_cells_index_list


def create_root_structure(root_cells: list, sorted_cells: list, tolerance: int) -> list:
    """
    Create list with all root cells and add (row, column) indexes to each

    :param root_cells: list with indexes of root boxes
    :param sorted_cells: list with all cells
    :param tolerance: int value representing maximal allowed distance between cells to be still considered as neighbours
    :return: list with root structure, contains all cells and (row, column) indexes for each
    """
    # now we should loop through root_boxes to obtain order of tables from top to
    # bottom or left to right
    root_cell_list = []
    for root_index, root_set in enumerate(root_cells):
        root_cell_list.append((sorted_cells[root_set]))

    return table_locator.find_rows_columns(root_cell_list, tolerance)


def create_table_structure(table_cell_index_list: list, sorted_cells: list, tolerance: int) -> list:
    """
    Create list with first level table structures, each cell have (row, column) indexes

    :param table_cell_index_list: list with indexes of first level boxes
    :param sorted_cells: list with all cells
    :param tolerance: int value representing maximal allowed distance
    between boxes to be still considered as neighbours.
    :return: list with first level table structures, contains all cells and (row, column) indexes for each
    """
    cell_list = []
    for table_index, indexes in enumerate(table_cell_index_list):
        cell_list.append([])
        for cell_index in indexes:
            cell_list[table_index].append(sorted_cells[cell_index])

    table_structure = []
    for cells in cell_list:
        table_structure.append(table_locator.find_rows_columns(cells, tolerance))

    return table_structure


def map_ocr_to_structure(cell_differences: dict, ocr_table_dict: dict, table_structure: list) -> list:
    """
    Map values obtained from ocr to first and second level table structures

    :param cell_differences: dict with cell differences
    :param ocr_table_dict: ocr text values in dict format
    :param table_structure: list with table index structures
    :return: list of tables fond in image
    """
    map_ocr_dict = {}
    # map_ocr_list = []
    for cell_index in ocr_table_dict:
        for ocr_index, ocr_value in enumerate(ocr_table_dict[cell_index]):
            cell = cell_differences[cell_index][ocr_index]
            map_ocr_dict[cell] = table_classes.TableCell(cell.x, cell.y, cell.width, cell.height, ocr_value)

    table_list = []
    for index_table, table in enumerate(table_structure):
        current_cells = []
        for cell, row_col in table:
            row, col = row_col
            table_cell = map_ocr_dict.setdefault(cell)
            if table_cell:
                table_cell.row = row
                table_cell.col = col
            else:
                table_cell = table_classes.TableCell(cell.x, cell.y, cell.width, cell.height, None, row, col)
            current_cells.append(table_cell)
        table_cell_list = sorted(current_cells, key=lambda tbl_cell: (tbl_cell.row, tbl_cell.col))
        table_list.append(table_classes.Table(table_cell_list))

    return table_list


def read_tables(super_sets: dict, sorted_cells: list, cell_differences: dict, ocr_table_dict: dict,
                tolerance: int) -> list:
    """
    Find and read tables on page. Return two level structure, second level is present
    only in case when we have nested table like structure

    :param super_sets: dict with supersets
    :param sorted_cells: list with all boxes
    :param cell_differences: dict with box differences
    :param ocr_table_dict: ocr text values in dict format
    :param tolerance: int value representing maximal allowed distance between cells to be still considered as neighbours
    :return: list with tables
    """
    root_list = find_root_cells(super_sets)
    sorted_cells, super_sets = expand_supersets(cell_differences, sorted_cells, super_sets, tolerance)
    table_cells_list = find_table_cells(super_sets, root_list, sorted_cells, tolerance)
    table_structure = create_table_structure(table_cells_list, sorted_cells, tolerance)
    table_list = map_ocr_to_structure(cell_differences, ocr_table_dict, table_structure)

    return table_list


def table_reader(filepath: str, tolerance_perc: float = 0.005, iterations: int = None) -> list:
    """
    Get list of tables from image.

    :param filepath: string with path to file.
    :param tolerance_perc: tolerance of computer vision, in percentage of image height pixels.
    :param iterations: number of iterations used for of dilation and erosion of image to extract table contours.
    :return: list of class Table objects, if any table present in given image.
    """
    sorted_boxes_list, super_set_boxes, img = table_locator.locate_table(filepath, tolerance_perc, iterations)

    h, _ = img.shape
    tolerance = round(h * tolerance_perc)

    differences = table_locator.supersets_differences(sorted_boxes_list, super_set_boxes, tolerance)
    preprocessed_image = table_ocr.image_preprocessing(img)
    ocr_results = table_ocr.table_ocr(preprocessed_image, differences, tolerance)

    table_list = read_tables(super_set_boxes, sorted_boxes_list, differences, ocr_results, tolerance)

    return table_list


if __name__ == "__main__":

    input_file = "data/documents/paper.pdf"
    zoom = 1.2
    page_num = 8
    img_dpi = 600

    # save image
    with fitz.open(input_file) as doc:
        mat = fitz.Matrix(zoom, zoom)
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat, dpi=img_dpi)
        output = f'data/page_images/page_{page.number}.jpg'
        pix.pil_save(output)

    # let's try it
    path = f'data/page_images/page_{page_num}.jpg'

    # sorted_boxes_list, super_set_boxes, image = table_locator.locate_table(path, tol, iters)
    # differences = table_locator.supersets_differences(sorted_boxes_list, super_set_boxes, tol)
    # preprocessed_image = table_ocr.image_preprocessing(image)
    # ocr_results = table_ocr.table_ocr(preprocessed_image, differences, tol)
    #
    # tables = read_tables(super_set_boxes, sorted_boxes_list, differences, ocr_results, tol)

    tables = table_reader(path)

    from tabulate import tabulate

    print("Tables:")
    for tbl in tables:
        if tbl:
            print(tabulate(tbl.table_to_dataframe(True), headers='keys', tablefmt='psql'))
