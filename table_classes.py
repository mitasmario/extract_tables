import pandas as pd


class Cell:
    """
    This class defines a cell as 2 dimensional rectangular object.
    """
    def __init__(self, x: float, y: float, width: float, height: float):
        """
        Initialization function of Cell class.

        :param x: coordinate x of top left corner of cell.
        :param y: coordinate y of top left corner of cell.
        :param width: cell width.
        :param height: cell height.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def print_cell(self) -> None:
        """
        Print cell attributes.

        :return: nothing
        """
        print(f"[x: {self.x}, y: {self.y}, width: {self.width}, height: {self.height}]")

    def cell_difference(self, other_cell, tolerance: int) -> dict:
        """
        Get box which represents part of this TableCell surface which don't overlap
        another TableCell.

        :param other_cell: another TableCell object to compare with this TableCell.
        :param tolerance: int, maximal width of resulting box which is considered as insignificant. For example if
                          tolerance = 15 and result is box with dimensions 15x100 we return 0x0 box as result.
        :return: dict with cells representing not overlapping surface from each side.
        """
        # checking not overlapping difference from left side
        if other_cell.x - self.x > tolerance:
            left = Cell(self.x,
                        max(self.y, other_cell.y),
                        other_cell.x - self.x,
                        min(self.y + self.height, other_cell.y + other_cell.height) - max(self.y, other_cell.y))
        else:
            left = None

        # check right side
        if self.x + self.width - other_cell.x - other_cell.width > tolerance:
            right = Cell(other_cell.x + other_cell.width,
                         max(self.y, other_cell.y),
                         self.x + self.width - other_cell.x - other_cell.width,
                         min(self.y + self.height, other_cell.y + other_cell.height) - max(self.y, other_cell.y))
        else:
            right = None

        # check top
        if other_cell.y - self.y > tolerance:
            top = Cell(self.x,
                       self.y,
                       self.width,
                       other_cell.y - self.y)
        else:
            top = None

        # check bottom
        if self.y + self.height - other_cell.y - other_cell.height > tolerance:
            bottom = Cell(self.x,
                          other_cell.y + other_cell.height,
                          self.width,
                          self.y + self.height - other_cell.y - other_cell.height)
        else:
            bottom = None

        return {"left": left, "right": right, "top": top, "bottom": bottom}

    def merge_2_boxes(self, other_cell, tolerance: int):
        """
        Create new cell by merging this cell with another.

        :param other_cell: another Cell object with 4 defined attributes - x, y, width, height.
        :param tolerance: int value representing maximal allowed distance between cells to be still considered as
               neighbours.
        :return: merged cell for merged box
        """
        # we have to identify which cell is holding top left corner of new merged cell and then just calculate width and
        # height of new cell
        if abs(self.x - other_cell.x) <= tolerance:
            if self.y < other_cell.y:
                first_box = self
                second_box = other_cell
            else:
                first_box = other_cell
                second_box = self
        else:
            if self.x < other_cell.x:
                first_box = self
                second_box = other_cell
            else:
                first_box = other_cell
                second_box = self

        new_x = min(first_box.x, second_box.x)
        new_y = min(first_box.y, second_box.y)
        new_width = max(first_box.x + first_box.width, second_box.x + second_box.width) - new_x
        new_height = max(first_box.y + first_box.height, second_box.y + second_box.height) - new_y

        return Cell(new_x, new_y, new_width, new_height)


class TableCell(Cell):
    """
    This class defines properties of single table cell
    """

    def __init__(self, x: float, y: float, width: float, height: float, content=None, row: int = None, col: int = None):
        """
        Initialization function of class TableCell. This class contains all needed information about particular
        cell in table.

        :param x: coordinate x of top left corner of cell.
        :param y: coordinate y of top left corner of cell.
        :param width: cell width.
        :param height: cell height.
        :param content: cell content.
        :param row: row index where cell is located in table
        :param col: column index where cell is located in table
        """
        super().__init__(x, y, width, height)
        self.content = content
        self.row = row
        self.col = col

    def print_cell(self) -> None:
        # super().print_cell()
        print(f"[x: {self.x}, y: {self.y}, width: {self.width}, height: {self.height}, "
              f"row: {self.row}, col: {self.col}]")
        print(f"Content:\n{self.content}")


class Table:
    """
    This class defines table as list of TableCells
    """

    def __init__(self, content):
        """
        Initialization function of class Table.

        :param content: list with TableCell objects.
        """
        self.content = content

    def table_to_dict(self, header: bool = True) -> dict:
        """
        Create dict from table.

        :param header: boolean parameter, True if first row represents header.
        :return: dict with table by columns.
        """
        column_list = []
        row_list = []
        for cell in self.content:
            if cell.row == 1:
                column_list.append(cell.content)
            if cell.col == 1:
                row_list.append(cell.content)

        sorted_table = sorted(self.content, key=lambda tbl_cell: (tbl_cell.col, tbl_cell.row))

        table_dict = {}
        for col_index in range(len(column_list)):
            n_row = len(row_list)
            col_cell_list = sorted_table[(col_index * n_row):(col_index * n_row + n_row)]
            cell_content_list = []
            for cell in col_cell_list:
                if not header:
                    cell_content_list.append(cell.content)
                elif header and cell.row != 1:
                    cell_content_list.append(cell.content)

            if header:
                colname = column_list[col_index]
            else:
                colname = f"column_{col_index}"
            table_dict[colname] = cell_content_list

        return table_dict

    def table_to_dataframe(self, header: bool = True) -> pd.DataFrame:
        """
        Create DataFrame from extracted table.

        :param header: boolean parameter, True if first row represents header.
        :return: DataFrame with table.
        """
        return pd.DataFrame(data=self.table_to_dict(header))
