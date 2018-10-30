import os

discussion_dir = 'discussions'

class DiscussionTree:

    def __init__(self, text, children):
        self.is_pro = text.startswith('Pro: ')

        if not self.is_pro and not text.startswith('Con: '):
            self.text = text
        else:
            self.text = text[len('Pro: '):]

        self.pro_children = []
        self.con_children = []

        for child in children:
            if child.is_pro:
                self.pro_children.append(child)
            else:
                self.con_children.append(child)

# lines comes in as a list of lines in the discussion
def build_discussion_dict(lines):
    cleaned_lines = []

    for line in lines:
        cleaned_lines.append(line.replace('\n', ''))

    lines = cleaned_lines

    discussion_tree = [lines[0][len('Discussion Title: '):], {}]

    for line in lines:
        # If the line doesn't start with a number, discard it
        if line == '' or not line[0].isdigit():
            continue

        number, text = line.split(' ', 1)

        current_tree = discussion_tree
        for value in number.split('.'):
            if value == '':
                continue
            value = int(value)

            if value not in current_tree[1]:
                current_tree[1][value] = [text, {}]

            current_tree = current_tree[1][value]

    return discussion_tree

def tree_to_discussion(discussion_tree):
    text = discussion_tree[0]
    children = discussion_tree[1].values()
    child_trees = []
    for child in children:
        child_trees.append(tree_to_discussion(child))
    discussion = DiscussionTree(text, child_trees)
    return discussion

for filename in os.listdir(discussion_dir):
    with open(os.path.join(discussion_dir, filename), 'r') as current_file:
        tree = build_discussion_dict(current_file.readlines())
        discussion = tree_to_discussion(tree)
