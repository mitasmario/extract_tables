import pytesseract
import cv2
import numpy as np
import table_locator
import re

# this is local path to tesseract installation folder, unfortunately on another computer this
# will be different, so please find path where you have installed tesseract and copy it here
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.3.0_1/bin/tesseract'


def image_preprocessing(img_bin: np.ndarray) -> np.array:
    """
    Pre-process image before ocr. Remove vertical and horizontal lines to prevent
    inference with ocr reading of text.

    :param img_bin: image on which ocr will be performed
    :return: processed image ready to be used in ocr part of process
    """
    result = img_bin.copy()
    thresh = cv2.threshold(img_bin, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # horizontal and vertical lines from table could interfere with ocr result.
    # we could end up with extra characters like | in text. Removing lines will
    # help to prevent it.

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    remove_horizontal = cv2.morphologyEx(thresh,
                                         cv2.MORPH_OPEN,
                                         horizontal_kernel,
                                         iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    remove_vertical = cv2.morphologyEx(thresh,
                                       cv2.MORPH_OPEN,
                                       vertical_kernel,
                                       iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 5)
    image = np.array(result)

    return image


def table_ocr(img: np.array, cell_differences: dict, table_cell_height_limit: int = 100, tolerance: int = 15,
              replace_newline: bool = True) -> dict:
    """

    :param img:
    :param cell_differences:
    :param table_cell_height_limit:
    :param tolerance:
    :param replace_newline:
    :return:
    """
    table_dict = {}
    for cell_set_index in cell_differences:
        cell_content = []
        for index, cell in enumerate(cell_differences[cell_set_index]):
            # extend area for ocr just by little bit to ensure that whole content of cell will be read
            y, x, w, h = cell.x - tolerance, cell.y - tolerance, cell.width + tolerance, cell.height + tolerance
            processed_image = img[x:x+h, y:y+w]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            # this function will have to cope with many formats of tables
            # sometimes cells could contain whole paragraph of text but
            # in other cases could contain just one letter or number.
            # Tesseract allows different configurations which has impact
            # on ocr result. Differentiation to different cases will help
            # to improve results. For now we just differentiate between
            # big and small cells, big cells will be read with page segmentation
            # appropriate for whole text reading. Small cells will be read
            # with column or one word page segmentation.
            if h > table_cell_height_limit:
                processed_image = cv2.dilate(processed_image, kernel, iterations=1)
                processed_image = cv2.erode(processed_image, kernel, iterations=1)
                custom_config = r'-c preserve_interword_spaces=1 --oem 2 --psm 3 -l eng+ita'
            else:
                processed_image = cv2.resize(processed_image, None, fx=1, fy=1.5, interpolation=cv2.INTER_CUBIC)
                processed_image = cv2.dilate(processed_image, kernel, iterations=1)
                processed_image = cv2.erode(processed_image, kernel, iterations=1)
                custom_config = r'-c preserve_interword_spaces=1 --oem 0 --psm 6 -l eng'
            selected = processed_image
            out = pytesseract.image_to_string(selected, config=custom_config)
            if replace_newline:
                out = out.replace("\n", " ")
            cell_content.append(out)

        table_dict[cell_set_index] = cell_content.copy()

    return table_dict


if __name__ == "__main__":

    # let's try it
    page_num = 5
    tol = 15
    path = f'data/page_images/{page_num}.jpg'

    sorted_cells_list, super_set_cells, image = table_locator.locate_table(path, tol)
    differences = table_locator.supersets_differences(sorted_cells_list, super_set_cells, 15)

    preprocessed_image = image_preprocessing(image)
    ocr_results = table_ocr(preprocessed_image, differences)
    print(ocr_results)