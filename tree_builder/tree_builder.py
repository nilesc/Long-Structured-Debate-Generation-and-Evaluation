import os

discussion_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'english_discussions')
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'input_files')

class DiscussionTree:

    def __init__(self, text, children):

        full_value, self.text = text.split(' ', 1)
        all_values = [value.strip('.') for value in full_value.split('.')]
        cleaned = [value for value in all_values if value != '']
        self.value = cleaned[-1]

        self.is_pro = not self.text.startswith('Con: ')
        if self.text.startswith('Con: ') or self.text.startswith('Pro: '):
            self.text = self.text[len('Pro: '):]

        self.children = {}

        for child in children:
            self.children[child.value] = child

    def get_pro_children(self):
        return [child for child in self.children.values() if child.is_pro]

    def get_con_children(self):
        return [child for child in self.children.values() if not child.is_pro]

    def get_children(self):
        return [child for child in self.children.values()]

    def get_arguments(self, pro=None):
        child_arguments = self.get_children()

        if pro:
            child_arguments = self.get_pro_children()
        if not pro and pro is not None:
            child_arguments = self.get_con_children()

        if not child_arguments:
            return [[self.text]]

        child_arguments = []
        for child in self.get_pro_children():
            child_arguments += child.get_arguments(pro)

        return [[self.text] + child for child in child_arguments]

    def fix_references(self, parent=None, root=None):
        if root is None:
            root = self

        if self.text.startswith('->'):
            if not 'discussion' in self.text:
                number = self.text.split(' ')[-1]

                current_root = root
                cleaned_values = [val for val in number.split('.')[1:] if val != '']
                for value in cleaned_values:
                    if value == '':
                        continue
                    current_root = current_root.children[value]

                parent.children[self.value] = current_root

        for child in self.children.values():
            child.fix_references(self, root)


# lines comes in as a list of lines in the discussion
def build_discussion_dict(lines):
    # print(lines)
    cleaned_lines = []

    for line in lines:
        if not line.startswith('1.'):
            continue
        cleaned_lines.append(line.replace('\n', ''))

    lines = cleaned_lines
    print(cleaned_lines)

    discussion_tree = None

    for line in lines:
        # If the line doesn't start with a number, discard it
        number, _ = line.split(' ', 1)

        if discussion_tree is None:
            discussion_tree = [line, {}]
            continue

        cleaned_values = [val for val in number.split('.')[1:] if val != '']

        current_tree = discussion_tree
        for value in cleaned_values:
            if value == '':
                continue
            value = int(value)

            if value not in current_tree[1]:
                current_tree[1][value] = [line, {}]

            current_tree = current_tree[1][value]

        print(discussion_tree)
        print()
    return discussion_tree

def tree_to_discussion(discussion_tree):
    text = discussion_tree[0]
    children = discussion_tree[1].values()
    child_trees = []
    for child in children:
        child_trees.append(tree_to_discussion(child))
    discussion = DiscussionTree(text, child_trees)
    return discussion

if __name__ == '__main__':
    for filename in os.listdir(discussion_dir)[0:1]:
        with open(os.path.join(discussion_dir, filename), 'r') as current_file:
            tree = build_discussion_dict(current_file.readlines())
            discussion = tree_to_discussion(tree)
            discussion.fix_references()
            args = discussion.get_arguments(pro=True)
            print(args)

            with open(os.path.join(output_dir, 'test.kialo_source'), 'w+') as source_file:
                for _ in args:
                    source_file.write(discussion.text + '\n')
