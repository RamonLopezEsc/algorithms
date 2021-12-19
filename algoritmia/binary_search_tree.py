#!/usr/bin/env python
# -*- coding: utf-8 -*-

class BinarySearchTree():
    def __init__(self):
        self.root = None

    def addNode(self, node):
        if self.root is None:
            self.root = node
        else:
            self._addNode(node, self.root)

    def _addNode(self, node, parent):
        if node < parent:
            if parent.leftChild != None:
                self._addNode(node, parent.leftChild)
            else:
                parent.leftChild = node
                parent.leftChild.parent = parent
        elif node > parent:
            if parent.rightChild != None:
                self._addNode(node, parent.rightChild)
            else:
                parent.rightChild = node
                parent.rightChild.parent = parent
        elif node == parent:
            parent.functionsValues.append(node.functionsValues)

    def stringify(self):
        """Return the string representation of the binary tree.

        :param bt: the binary tree
        :return: the string representation
        """
        if self is None:
            return ''
        return '\n' + '\n'.join(self.build_str(self.root)[0])

    def build_str(self, node):
        """Recursive function used for pretty-printing the binary tree.

        In each recursive call, a "box" of characters visually representing the
        current subtree is constructed line by line. Each line is padded with
        whitespaces to ensure all lines have the same length. The box, its width,
        and the start-end positions of its root (used for drawing branches) are
        sent up to the parent call, which then combines left and right sub-boxes
        to build a bigger box etc.
        """
        if node is None:
            return [], 0, 0, 0

        line1 = []
        line2 = []
        new_root_width = gap_size = len(str(node.functionsValues))

        # Get the left and right sub-boxes, their widths and their root positions
        l_box, l_box_width, l_root_start, l_root_end = self.build_str(node.leftChild)
        r_box, r_box_width, r_root_start, r_root_end = self.build_str(node.rightChild)

        # Draw the branch connecting the new root to the left sub-box,
        # padding with whitespaces where necessary
        if l_box_width > 0:
            l_root = -int(-(l_root_start + l_root_end) / 2) + 1  # ceiling
            line1.append(' ' * (l_root + 1))
            line1.append('_' * (l_box_width - l_root))
            line2.append(' ' * l_root + '/')
            line2.append(' ' * (l_box_width - l_root))
            new_root_start = l_box_width + 1
            gap_size += 1
        else:
            new_root_start = 0

        # Draw the representation of the new root
        line1.append(str(node.functionsValues))
        line2.append(' ' * new_root_width)

        # Draw the branch connecting the new root to the right sub-box,
        # padding with whitespaces where necessary
        if r_box_width > 0:
            r_root = int((r_root_start + r_root_end) / 2)  # floor
            line1.append('_' * r_root)
            line1.append(' ' * (r_box_width - r_root + 1))
            line2.append(' ' * r_root + '\\')
            line2.append(' ' * (r_box_width - r_root))
            gap_size += 1
        new_root_end = new_root_start + new_root_width - 1

        # Combine the left and right sub-boxes with the branches drawn above
        gap = ' ' * gap_size
        new_box = [''.join(line1), ''.join(line2)]
        for i in range(max(len(l_box), len(r_box))):
            l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
            r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
            new_box.append(l_line + gap + r_line)

        # Return the new box, its width and its root positions
        return new_box, len(new_box[0]), new_root_start, new_root_end

    def pprint(self):
        """Pretty print the binary tree.

        :param bt: the binary tree to pretty print
        :raises ValueError: if an invalid tree is given
        """
        print(self.stringify())

class TreeNode():
    def __init__(self, value):
        if type(value) != list:
            raise ValueError("Value is not list instance")
        self.functionsValues = [ ]
        self.functionsValues.append(value)
        self.parent = None
        self.rightChild = None
        self.leftChild = None

    def __lt__(self, node):
        if self.functionsValues[0][0] <= node.functionsValues[0][0] and\
           self.functionsValues[0][1] >  node.functionsValues[0][1]:
            return True
        elif self.functionsValues[0][0] <  node.functionsValues[0][0] and\
             self.functionsValues[0][1] >= node.functionsValues[0][1]:
            return True
        else:
            return False

    def __gt__(self, node):
        if self.functionsValues[0][0] >= node.functionsValues[0][0] and\
           self.functionsValues[0][1] <  node.functionsValues[0][1]:
            return True
        elif self.functionsValues[0][0] > node.functionsValues[0][0] and\
             self.functionsValues[0][1] <= node.functionsValues[0][1]:
            return True
        else:
            return False

    def __eq__(self, node):
        if self.functionsValues[0][0] < node.functionsValues[0][0] and\
           self.functionsValues[0][1] <= node.functionsValues[0][1]:
            return True
        elif self.functionsValues[0][0] >= node.functionsValues[0][0] and\
           self.functionsValues[0][1] > node.functionsValues[0][1]:
            return True
        else:
            return False

bst = BinarySearchTree()

# node1 = TreeNode([12, 1])
# node2 = TreeNode([16, 2])
#
# print node1 == node2
