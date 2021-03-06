import os
import re
import progressbar
import ner
from itertools import tee

discussion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/filtered_discussions')
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/input_files')
end_of_argument = '<EOA>'

class InvalidResponseType(Exception):
    pass

class DiscussionTree:

    def __init__(self, text, children):
        full_value, self.text = text.split(' ', 1)
        all_values = [value.strip('.') for value in full_value.split('.')]
        cleaned = [value for value in all_values if value != '']
        self.position = cleaned[0:-1]
        self.value = cleaned[-1]

        self.is_pro = not self.text.startswith('Con: ')
        if self.text.startswith('Con: ') or self.text.startswith('Pro: '):
            self.text = self.text[len('Pro: '):]

        if not self.text.endswith('.'):
            self.text += '.'

        self.children = {}

        for child in children:
            self.children[child.value] = child

    def print_pre_order(self, tabs=''):
        '''
        Prints pre-order traversal of debate tree
        '''
        print(tabs + self.text)
        print("--")
        for child in self.children.values():
            child.print_pre_order(tabs+'    ')

    def get_pro_children(self):
        return [child for child in self.children.values() if child.is_pro]

    def get_con_children(self):
        return [child for child in self.children.values() if not child.is_pro]

    def get_children(self):
        return [child for child in self.children.values()]

    def get_arguments(self, pro=None, augmentor=None):
        base_args = self.get_arguments_inner(pro)

        if augmentor:
            base_args = augmentor(base_args)

        return base_args

    def build_args(self, response_type='All'):
        '''
        Function that builds all args of the type specified by response_type
        @response_type ('All', 'Pro', or 'Con'):
            All: Return all valid prompt / response pairings in the debate tree
            Pro: Return all valid prompts paired with supporting responses
            Con: Return all valid propts paired with contradicting responses
        '''
        # Build all the paths in the tree that lead to leaves
        paths = self.build_all_paths()
        parsed_args = []
        for path in paths:
            # Get all the valid argument pairings for this path
            path_parsed_args = build_path_parsed_args(path, response_type)
            parsed_args.extend(path_parsed_args)

        return remove_duplicates(parsed_args)

    def build_all_paths(self):
        '''
        Function that builds all the paths in the tree that lead to leaves
        '''
        paths = []
        for child in self.children.values():
            paths.extend(child.build_all_paths())

        complex_arg = (self.text, self.is_pro)

        # Add all the paths from root to leaf for this tree
        paths.extend(self.build_paths([complex_arg], path_cons=0, path_depth=1))
        return paths

    def build_paths(self, path, path_cons, path_depth):
        '''
        Function that builds all the paths in the tree from root to leaf
        @path: The path recursively built so far
        @path_cons: The number of cons encountered in the path
        @path_depth: The length of the path
        '''
        paths = []

        # Base condition: If at a leaf with a longer than 1 path depth, return built path
        if not self.children.values() and path_depth > 1:
            return [path]

        # Add all children to the built path
        for child in self.children.values():
            complex_arg = (child.text, child.is_pro)
            built_path = path + [complex_arg]
            built_path_cons = path_cons
            built_path_depth = path_depth + 1
            if not child.is_pro:
                built_path_cons += 1
             # Optimization: ignore paths that have 3 or more cons
            if built_path_cons < 3:
                paths.extend(child.build_paths(built_path, built_path_cons,
                    built_path_depth))
            # Only paths that are longer than one will have prompt, response pairings
            elif built_path_depth > 1:
                paths.append(built_path)

        return paths


    def build_complex_args(self, pro_responses=None, chain_responses=True):
        """
        Builds arguments according to the regex [Pro|Con][Pro]*[Pro|Con][Pro]*.
        If pro_responses is True, then the regex will instead be [Pro|Con][Pro]*
        If pro_responses is False, then the regex will instead be [Pro|Con][Pro]*[Con][Pro]*.
        """
        unparsed_args = self.build_complex_args_inner(chain_responses)
        parsed_args = []

        if chain_responses:
            all_responses = []
            for arg in unparsed_args:
                sentences, is_pro = zip(*arg)
                all_responses.append(DiscussionTree.split_at_cons(sentences, is_pro))

            # Front augmentation is necessary to make sure we have all valid chains
            augmented = front_augmentation(all_responses)
            augmented = [x for x in augmented if len(x) != 1]

            return remove_duplicates(augmented)

        for arg in unparsed_args:
            if len(arg) == 1:
                continue

            sentences, is_pro = zip(*arg)

            # Look at all subsets that begin at the start
            for slice_index in range(2, len(sentences)+1):
                reduced_sentences = sentences[:slice_index]
                reduced_is_pro = is_pro[:slice_index]

                # If there are no con arguments other than first, then add all slices
                # Don't include this if pro_responses is False
                if all(reduced_is_pro[1:]) and not (pro_responses != None and not pro_responses):
                    parsed_args.extend(slice_augmentation([list(reduced_sentences)]))

                # If there is a con argument, then split at that point
                # Don't include this if pro_responses is True
                elif not all(reduced_is_pro[1:]) and not pro_responses:
                    split_point = reduced_is_pro[1:].index(False) + 1
                    first_part = sentences[:split_point]
                    second_part = sentences[split_point:]

                    first_part_condensed = ' '.join(first_part)

                    for crop_point in range(len(second_part)):
                        parsed_args.append([first_part_condensed] + list(second_part)[:crop_point+1])

        return remove_duplicates(parsed_args)


    @classmethod
    def split_at_cons(cls, args, is_pro):
        def pairwise_iter(iterable):
            it = iter(iterable)
            a, b = tee(it)
            next(b, None)
            return zip(a, b)

        # If our first point is a pro, we still want to start an argument at it
        is_pro = list(is_pro)
        is_pro[0] = False
        indices = [i for i, x in enumerate(is_pro) if not x] + [0]

        split = []

        for pair in pairwise_iter(indices):
            portion = args[pair[0]:pair[1] or None]
            split.append(''.join(portion))

        return split


    # For each node in our tree, call traverse_complex and append its own text
    # and is_pro value to the resulting arguments
    def build_complex_args_inner(self, chain_responses):
        all_args = []

        children = self.get_children()

        # for every node in tree
        for child in children:
            child_args = child.build_complex_args_inner(chain_responses)
            all_args.extend(child_args)

            for complex_arg in child.traverse_complex(False, chain_responses):
                all_args.append([(self.text, self.is_pro)] + complex_arg)

        return all_args

    def traverse_complex(self, seen_con, chain_responses):
        # If we have already seen a con argument and we see another
        seen_con = seen_con or not self.is_pro

        children = []

        if not seen_con or chain_responses:
            children = self.get_children()
        else:
            children = self.get_pro_children()

        if not children:
            return [[(self.text, self.is_pro)]]

        partial_args = []
        for child in children:
            partial_args.extend(child.traverse_complex(seen_con, chain_responses))

        return_list = [[(self.text, self.is_pro)] + partial_arg for partial_arg in partial_args]

        return return_list


    def get_arguments_inner(self, pro):
        children = self.get_children()

        if pro:
            children = self.get_pro_children()
        if not pro and pro is not None:
            children = self.get_con_children()

        if not children:
            return [[self.text]]

        child_arguments = []

        for child in children:
            child_arguments += child.get_arguments_inner(pro)

        all_args = [[self.text] + child for child in child_arguments]

        return all_args

    def fix_references(self, parent=None, root=None):
        if root is None:
            root = self

        if self.text.startswith('->'):
            if 'discussion' in self.text:
                self.text = ': '.join(self.text.split(': ')[1:])
            else:
                number = self.text.split(' ')[-1]

                current_root = root
                cleaned_values = [val for val in number.split('.')[1:] if val != '']
                for value in cleaned_values:
                    if value == '':
                        continue
                    current_root = current_root.children[value]

                parent.children[self.value].text = current_root.text

        for child in self.children.values():
            child.fix_references(self, root)

    def clean_named_entities(self, root=True):
        # Root nodes are often formatted with capitalization, which messes up the NER
        if not root:
            self.text = ner.replace_entities(self.text, replace_with=None)

        for child in self.get_children():
            child.clean_named_entities(root=False)


def build_path_parsed_args(path, response_type):
    '''
    Helper function that generates all the valid prompt response pairings in a given path.
    @path: Path in debate tree
    @response_type: 'All,' 'Pro,' or 'Con': what dataset to generate
    '''
    parsed_args = []
    prompt = []
    # For all valid prompts ranging from the start to one before end of the list
    for i in range(len(path)-1):
        is_first = i == 0
        arg, is_pro = path[i]
        if is_first or is_pro:
            # Valid prompt encountered, add all valid responses
            prompt = prompt + [arg]
            responses = build_path_responses(path, i, response_type)
            pairings = [(' '.join(prompt), ' '.join(response)) for response in responses]
            parsed_args.extend(pairings)
        else:
            # Stop generating new prompts once invalid prompt encountered
            break

    return parsed_args

def build_path_responses(path, position, response_type):
    '''
    Helper function that generates all valid responses starting at a position in a path.
    @path: Path in debate tree
    @position: Position in the path
    @response_type ('All,' 'Pro,' or 'Con'): What dataset to generate
    '''
    responses = []
    response = []

     # For all potentially valid responses that start at the end of the prompt
    for j in range(position+1, len(path)):
        is_first_response = j == position + 1
        argument, is_pro_response = path[j]

        if can_append(is_first_response, is_pro_response, response_type):
            response = response + [argument]
            responses.append(response)
        else:
            # Stop appending responses once invalid response encountered
            break

    return responses

def can_append(is_first_response, is_pro_response, response_type):
    '''
    Helper function to evaluate whether a reponse can be appended
    @is_first_response: Whether it's the first response
    @is_pro_response: Whether it is a supporting response
    @response_type ('All,' 'Pro,' or 'Con'): What dataset to generate
    '''
    if response_type == 'All':
        return is_first_response or is_pro_response
    elif response_type == 'Pro':
        return is_pro_response
    elif response_type == 'Con':
        is_first_and_con = is_first_response and (not is_pro_response)
        is_next_and_pro = (not is_first_response) and is_pro_response
        return is_first_and_con or is_next_and_pro
    else:
        raise InvalidResponseType(f"{response_type} is not a valid response type.")

# Lines comes in as a list of lines in the discussion
def build_discussion_dict(lines):
    cleaned_lines = []

    for line in lines:
        if not line.startswith('1.'):
            continue
        prev_line = ''
        cleaned_line = line

        # Removes URLs from arguments
        while cleaned_line != prev_line:
            prev_line = cleaned_line
            cleaned_line = re.sub(r'(.*)\[(.*)\]\((.*)\)(.*)', r'\1\2\4', prev_line.rstrip())
        cleaned_lines.append(cleaned_line.replace('\n', ''))

    lines = cleaned_lines

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

    return discussion_tree


def front_augmentation(args):
    """
    This is a method of augmentation that says that, for a given prompt and response,
    the same prompt and the response without the last sentence is also a vaild prompt
    response pair. If we start with
        Prompt: A, Response: B C D
    where all letters are sentences, this method would add
        Prompt: A, Response: B C
        Prompt: A, Response: B
    """
    augmented = [arg for arg in args]
    for arg in args:
        for i in range(len(arg)-2):
            augmented.append(arg[:(i+2)])

    return augmented


def back_augmentation(args):
    """
    This is a method of augmentation that says that, for a given prompt and response,
    the first sentence of the response can be a valid prompt, and the rest of the
    response will be a response to this new prompt. If we start with
        Prompt A, Response: B C D
    where all letters are sentences, this method would add
        Prompt: B, Response: C D
        Prompt: C, Response: D
    """
    augmented = [arg for arg in args]
    for arg in args:
        for i in range(len(arg)-2):
            augmented.append(arg[(i+1):])

    return augmented

def slice_augmentation(args):
    """
    This is a method of augmentation that says that, for a given prompt and response,
    the prompt and any number of sentences from the start of the response can be a
    valid prompt, and the remainder of the response will be a valid response to this
    new prompt. If we start with
        Prompt A, Response: B C D
    where all letters are sentences, this method would add
        Prompt: A B, Response: C D
        Prompt: A B C, Response: D
    """
    augmented = []
    for arg in args:
        for i in range(len(arg) - 1):
            first_part = arg[:i+1]
            second_part = arg[i+1:]
            combined = ' '.join(first_part)
            augmented.append([combined] + second_part)
    return augmented


def total_augmentation(args):
    """
    Runs a combination of back_augmentation, front_augmentation, and slice_augmentation.
    This should return all viable arguments that can be made from any subset of the original
    argument.
    """
    args = back_augmentation(args)
    args = front_augmentation(args)
    args = slice_augmentation(args)

    return remove_duplicates(args)

def remove_duplicates(x):
    list_of_tuples = list(set(tuple(value) for value in x))
    return [list(value) for value in list_of_tuples]

def tree_to_discussion(discussion_tree):
    text = discussion_tree[0]
    children = discussion_tree[1].values()
    child_trees = []
    for child in children:
        child_trees.append(tree_to_discussion(child))
    discussion = DiscussionTree(text, child_trees)
    return discussion


def write_discussions_to_files(discussion_dir, filename, source_file, target_file):
    with open(os.path.join(discussion_dir, filename), 'r') as current_file:
        if not filename.endswith('txt'):
            return
        tree = build_discussion_dict(current_file.readlines())

        chain_responses = True
        discussion = tree_to_discussion(tree)
        discussion.fix_references()
        #discussion.clean_named_entities()
        #args = discussion.build_args(response_type = 'All')
        args = discussion.build_complex_args(pro_responses = None, chain_responses = chain_responses)
        #args = discussion.get_arguments(pro=True, augmentor=back_augmentation)

        for arg in args:
            if chain_responses:
                prompt = (' ' + end_of_argument + ' ').join(arg[:-1]) + ' ' + end_of_argument
                response = arg[-1]
            else:
                prompt = arg[0]
                response = ' '.join(arg[1:])

            source_file.write(prompt + '\n')
            target_file.write(response + '\n')


def write_source_target(discussion_dir, filenames, name, start, end):
    source_file = open(os.path.join(output_dir, f'{name}.kialo_source'), 'w+')
    target_file = open(os.path.join(output_dir, f'{name}.kialo_target'), 'w+')

    p = progressbar.ProgressBar(term_width=80)

    print(f'Extracting {name} arguments: ')
    for filename in p(filenames[start:end]):
        write_discussions_to_files(discussion_dir, filename, source_file, target_file)

    source_file.close()
    target_file.close()


if __name__ == '__main__':
    filenames = os.listdir(discussion_dir)
    test_start = 0
    test_end = int(len(filenames) * .05)
    valid_start = test_end
    valid_end = valid_start + int(len(filenames) * .1)
    train_start = valid_end
    train_end = len(filenames)

    write_source_target(discussion_dir, filenames, 'test', test_start, test_end)
    write_source_target(discussion_dir, filenames, 'valid', valid_start, valid_end)
    write_source_target(discussion_dir, filenames, 'train', train_start, train_end)
