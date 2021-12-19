#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Node:
    def __init__(self, val):
        self.izquierda = None
        self.derecha = None
        self.padre = None
        self.valor = val
        self.factorbalance = 0

    def isLeftChild(self):
        return self.padre and self.padre.izquierda == self

    def isRightChild(self):
        return self.padre and self.padre.derecha == self

    def isRoot(self):
        return not self.padre

class Range_Tree:
    def __init__(self):
        self.raiz = None

    def add(self, val):
        if(self.raiz == None):
            self.raiz = Node(val)
            self.raiz.izquierda = Node(val)
            self.raiz.izquierda.padre = self.raiz
            self.raiz.factorbalance = 1
        else:
            self._add(val, self.raiz)

    def _add(self, val, node):
        if(val < node.valor):
            if(node.izquierda != None):
                self._add(val, node.izquierda)
            else:
                node.izquierda = Node(val)
                node.izquierda.padre = node
                self.updateBalance(node.izquierda)
        else:
            if(node.derecha != None):
                self._add(val, node.derecha)
            else:
                node.derecha = Node(val)
                node.derecha.padre = node
                self.updateBalance(node.derecha)

    def updateBalance(self,node):
        if node.factorbalance > 1 or node.factorbalance < -1:
            self.rebalance(node)
            return
        if node.padre != None:
            if node.isLeftChild():
                    node.padre.factorbalance += 1
            elif node.isRightChild():
                    node.padre.factorbalance -= 1
            if node.padre.factorbalance != 0:
                    self.updateBalance(node.padre)

    def rotateLeft(self,rotRoot):
        newRoot = rotRoot.derecha
        rotRoot.derecha = newRoot.izquierda
        if newRoot.izquierda != None:
            newRoot.izquierda.padre = rotRoot
        newRoot.padre = rotRoot.padre
        if rotRoot.isRoot():
            self.raiz = newRoot
        else:
            if rotRoot.isLeftChild():
                    rotRoot.padre.izquierda = newRoot
            else:
                rotRoot.padre.derecha = newRoot
        newRoot.izquierda = rotRoot
        rotRoot.padre = newRoot
        rotRoot.factorbalance = rotRoot.factorbalance + 1 - min(newRoot.factorbalance, 0)
        newRoot.factorbalance = newRoot.factorbalance + 1 + max(rotRoot.factorbalance, 0)

    def rotateRight(self,rotRoot):
        newRoot = rotRoot.izquierda
        rotRoot.izquierda = newRoot.derecha
        if newRoot.derecha != None:
            newRoot.derecha.padre = rotRoot
        newRoot.padre = rotRoot.padre
        if rotRoot.isRoot():
            self.raiz = newRoot
        else:
            if rotRoot.isLeftChild():
                    rotRoot.padre.izquierda = newRoot
            else:
                rotRoot.padre.derecha = newRoot
        newRoot.r = rotRoot
        rotRoot.padre = newRoot
        rotRoot.factorbalance = rotRoot.factorbalance - 1 - max(newRoot.factorbalance, 0)
        newRoot.factorbalance = newRoot.factorbalance - 1 + min(rotRoot.factorbalance, 0)

    def rebalance(self,node):
      if node.factorbalance < 0:
             if node.derecha.factorbalance > 0:
                self.rotateRight(node.derecha)
                self.rotateLeft(node)
             else:
                self.rotateLeft(node)
      elif node.factorbalance > 0:
             if node.izquierda.balanceFactor < 0:
                self.rotateLeft(node.izquierda)
                self.rotateRight(node)
             else:
                self.rotateRight(node)

    def tree_maximum(self, node):
        if node.derecha is not None:
            return self.tree_maximum(node.derecha)
        else:
            return node

    def build_from_list(self,list):
        sorted_list = sorted(list)
        self.raiz = self.divide_list(sorted_list)

    def divide_list(self, sorted_list):
        if len(sorted_list) <= 1:
            new_node = Node(sorted_list[0])
            return new_node
        else:
            left = self.divide_list(sorted_list[0:int(len(sorted_list)/2)])
            right = self.divide_list(sorted_list[int(len(sorted_list)/2):len(sorted_list)])
            new_node = Node(self.tree_maximum(left).valor)
            new_node.izquierda = left
            new_node.derecha = right
            return new_node

    def stringify(self):
        """Return the string representation of the binary tree.

        :param bt: the binary tree
        :return: the string representation
        """
        if self is None:
            return ''
        return '\n' + '\n'.join(self.build_str(self.raiz)[0])

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
        new_root_width = gap_size = len(str(node.valor))

        # Get the left and right sub-boxes, their widths and their root positions
        l_box, l_box_width, l_root_start, l_root_end = self.build_str(node.izquierda)
        r_box, r_box_width, r_root_start, r_root_end = self.build_str(node.derecha)

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
        line1.append(str(node.valor))
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