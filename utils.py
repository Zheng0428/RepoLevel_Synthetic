# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import ast
import json
import os
import random
import re
import subprocess
import uuid
from collections import OrderedDict
import logging, yaml, tiktoken
logger = logging.getLogger(__name__)
from envs import PLAYGROUND_DIR, PROJECT_FILE_LOC, DEFAULT_PATH


def show_project_structure(
    structure: dict,
    level: int = 0,
    indentation: int = 4,
    randomize: bool = False,
) -> str:
    """pprint the project structure, with randomization option"""

    pp_string = ""

    items = list(structure.items())
    if randomize:
        random.shuffle(items)

    for key, value in items:
        if "." in key and ".py" not in key:
            continue  # skip none python files

        pp_string += " " * (level * indentation) + str(key) + "\n"
        if (
            isinstance(value, dict) and "classes" not in value
        ):  # Check if value is a dictionary
            pp_string += show_project_structure(
                value, level + 1, indentation, randomize
            )

    return pp_string


def filter_out_test_files(structure: dict):
    """filter out test files from the project structure **in place**"""
    for key, value in list(structure.items()):
        if key.startswith("test"):
            del structure[key]
        elif isinstance(value, dict):
            filter_out_test_files(value)


def keep_test_files(structure: dict) -> dict:
    """Keep only test-related files from the project structure."""
    result: dict = {}
    for key, value in structure.items():
        if "test" in key.lower():
            result[key] = value
        elif isinstance(value, dict):
            # Recursively process subdirectories
            filtered_substructure = keep_test_files(value)
            if filtered_substructure:
                result[key] = filtered_substructure
    return result


def filter_none_python(structure: dict) -> dict:
    """NOTE: this is done in place"""
    for key, value in list(structure.items()):
        if (
            not "functions" in value.keys()
            and not "classes" in value.keys()
            and not "text" in value.keys()
        ) or not len(value.keys()) == 3:
            filter_none_python(value)

            if structure[key] == {}:
                del structure[key]
        else:
            if not key.endswith(".py"):
                del structure[key]


def get_full_file_paths_and_classes_and_functions(structure, current_path=""):
    """
    Recursively retrieve all file paths, classes, and functions within a directory structure.

    Arguments:
    structure -- a dictionary representing the directory structure
    current_path -- the path accumulated so far, used during recursion (default="")

    Returns:
    A tuple containing:
    - files: list of full file paths
    - classes: list of class details with file paths
    - functions: list of function details with file paths
    """
    files = []
    classes = []
    functions = []
    for name, content in structure.items():
        if isinstance(content, dict):
            if (
                not ("functions" in content.keys() and type(content["functions"]) == list)
                and not ("classes" in content.keys() and type(content["classes"]) == list)
                and not ("text" in content.keys() and type(content["text"]) == list)
            ) or not len(content.keys()) == 3:
                # or guards against case where functions and classes are somehow part of the structure.
                next_path = f"{current_path}/{name}" if current_path else name
                (
                    sub_files,
                    sub_classes,
                    sub_functions,
                ) = get_full_file_paths_and_classes_and_functions(content, next_path)
                files.extend(sub_files)
                classes.extend(sub_classes)
                functions.extend(sub_functions)
            else:
                next_path = f"{current_path}/{name}" if current_path else name
                if "text" in content:
                    files.append((next_path, content["text"]))
                if "classes" in content:
                    for clazz in content["classes"]:
                        classes.append(
                            {
                                "file": next_path,
                                "name": clazz["name"],
                                "start_line": clazz["start_line"],
                                "end_line": clazz["end_line"],
                                "methods": [
                                    {
                                        "name": method["name"],
                                        "start_line": method["start_line"],
                                        "end_line": method["end_line"],
                                    }
                                    for method in clazz.get("methods", [])
                                ],
                            }
                        )
                if "functions" in content:
                    if type(content["functions"]) == dict:  # TODO: why fuctions is a dict in some cases?
                        content["functions"] = [content["functions"]]
                    for function in content["functions"]:
                        function["file"] = next_path
                        functions.append(function)
        else:
            next_path = f"{current_path}/{name}" if current_path else name
            files.append(next_path)
    return files, classes, functions


def get_repo_detail(instance_id: str) -> dict:
    with open(os.path.join(PROJECT_FILE_LOC, instance_id + ".json")) as f:
        return json.load(f)


def get_repo_structure(instance_id: str) -> dict:
    return get_repo_detail(instance_id)["structure"]

def get_file_contents(instance_id: str, pred_files: list[str]) -> dict[str, str]:
    structure = get_repo_structure(instance_id)
    repo_file_contents, _, _ = get_full_file_paths_and_classes_and_functions(
        structure
    )
    repo_file_contents_dict = {path: lines for path, lines in repo_file_contents}
    return {
        pred_file: "\n".join(repo_file_contents_dict[pred_file])
        for pred_file in pred_files
        # # This should be always true except for one special GT case:
        # # astropy/coordinates/builtin_frames/itrs_observed_transforms.py
        # # This is fixed in the GT file (12/26/24).
        # if pred_file in repo_file_contents_dict
    }

def get_repo_files(structure, filepaths: list[str]):
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    file_contents = dict()
    for filepath in filepaths:
        content = None

        for file_content in files:
            if file_content[0] == filepath:
                content = "\n".join(file_content[1])
                file_contents[filepath] = content
                break

        assert content is not None, "file not found"
    return file_contents


def correct_file_paths(model_found_files, files):
    found_files = []
    if model_found_files:
        for model_file in model_found_files:
            # Check if any model found file is a subset of the current file path
            for file_content in files:
                file = file_content[0]
                if model_file == file:
                    found_files.append(file)
        return found_files
    else:
        return []


def check_syntax(code):
    if not isinstance(code, list):
        code = [code]

    for c in code:
        if (
            not c.strip()
        ):  # Check for cases where the model didn't return a python block
            return False
        try:
            ast.parse(c)
        except SyntaxError as e:
            return False
    return True


def remove_empty_lines(code: str) -> str:
    # Split the code into lines
    lines = code.splitlines()
    # Remove empty lines
    filtered_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(filtered_lines)


def check_code_differ_by_just_empty_lines(codes, prev_codes) -> bool:

    if not isinstance(codes, list):
        codes = [codes]
        prev_codes = [prev_codes]

    normalized_code1 = ""
    normalized_code2 = ""

    for code, prev_code in zip(codes, prev_codes):
        # Normalize both code snippets
        normalized_code1 += remove_empty_lines(code)
        normalized_code2 += remove_empty_lines(prev_code)

    return normalized_code1 == normalized_code2


def lint_code(repo_playground, temp_name, code, prev_code="") -> tuple[bool, set, set]:

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    with open(f"{repo_playground}/{temp_name}", "w") as f:
        f.write(prev_code)

    # lint the code
    # check for fatal errors
    fatal = "E9,F821,F823,F831,F406,F407,F701,F702,F704,F706"
    o = subprocess.run(
        f"flake8 --select={fatal} --isolated {repo_playground}/{temp_name}",
        shell=True,
        capture_output=True,
    )
    s = o.stdout.decode("utf-8")

    prev_errors = set()
    if s != "":
        for error in s.split(f"{repo_playground}/{temp_name}:")[1:]:
            num_free_error = ":".join(error.split(":")[2:]).strip()
            prev_errors.add(num_free_error)

    with open(f"{repo_playground}/{temp_name}", "w") as f:
        f.write(code)

    o = subprocess.run(
        f"flake8 --select={fatal} --isolated {repo_playground}/{temp_name}",
        shell=True,
        capture_output=True,
    )
    s = o.stdout.decode("utf-8")

    # remove playground
    subprocess.run(f"rm -rf {repo_playground}", shell=True)

    errors = set()
    if s != "":
        for error in s.split(f"{repo_playground}/{temp_name}:")[1:]:
            num_free_error = ":".join(error.split(":")[2:]).strip()
            errors.add(num_free_error)

    if len(errors - prev_errors) > 0:
        return False, prev_errors, errors

    return True, set(), set()


def fake_git_repo(file_pathes, old_contents, new_contents, files_exist=True) -> str:
    """create a fake git repo to obtain git diff format
    
    Args:
        file_pathes: List of file paths or single file path
        old_contents: List of old file contents or single file content. Must be empty strings when files_exist=False
        new_contents: List of new file contents or single file content
        files_exist: Whether the files already exist in the repo. If False, will create new files
                    to generate 'new file mode' in git diff output
    """
    if old_contents == new_contents:
        print('error')
    if not isinstance(file_pathes, list):
        # for backwards compatibility
        file_pathes = [file_pathes]
        old_contents = [old_contents]
        new_contents = [new_contents]

    # Validate that old_contents are empty when files_exist=False
    if not files_exist:
        if any(content.strip() for content in old_contents):
            raise ValueError("old_contents must be empty strings when files_exist=False")

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(PLAYGROUND_DIR, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    # create a fake git repo
    subprocess.run(f"cd {repo_playground} && git init", shell=True)

    # If files exist, create initial commit with old content
    if files_exist:
        for file_path, old_content, new_content in zip(
            file_pathes, old_contents, new_contents
        ):
            # create a file
            subprocess.run(
                f"mkdir -p {repo_playground}/{os.path.dirname(file_path)}", shell=True
            )

            with open(f"{repo_playground}/{file_path}", "w") as f:
                f.write(old_content)

            # add file to git
            subprocess.run(
                f"cd {repo_playground} && git add {file_path} && git commit -m 'initial commit'",
                shell=True,
            )

        # Now write new content
        for file_path, old_content, new_content in zip(
            file_pathes, old_contents, new_contents
        ):
            with open(f"{repo_playground}/{file_path}", "w") as f:
                f.write(new_content)

        # Get diff for modified files
        o = subprocess.run(
            f"cd {repo_playground} && git diff", 
            shell=True, 
            capture_output=True
        )
    else:
        # For new files, create them with new content
        for file_path, new_content in zip(file_pathes, new_contents):
            # create directory if needed
            subprocess.run(
                f"mkdir -p {repo_playground}/{os.path.dirname(file_path)}", shell=True
            )
            
            # create new file
            with open(f"{repo_playground}/{file_path}", "w") as f:
                f.write(new_content)

        # Get diff for new files
        o = subprocess.run(
            f"cd {repo_playground} && git add . && git diff --cached", 
            shell=True, 
            capture_output=True
        )

    s = o.stdout.decode("utf-8")

    # remove playground
    subprocess.run(f"rm -rf {repo_playground}", shell=True)

    return s


def fake_git_apply(repo_playground, file_path, old_content, patch) -> str:
    """create a fake git repo to obtain new file content"""

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    # create a fake git repo
    subprocess.run(f"cd {repo_playground} && git init", shell=True)

    # create a file
    subprocess.run(
        f"mkdir -p {repo_playground}/{os.path.dirname(file_path)}", shell=True
    )

    with open(f"{repo_playground}/{file_path}", "w") as f:
        f.write(old_content)

    # add file to git
    subprocess.run(
        f"cd {repo_playground} && git add {file_path} && git commit -m 'initial commit'",
        shell=True,
    )

    # apply patch file
    patch_file = f"{str(uuid.uuid4())}.patch"
    with open(f"{repo_playground}/{patch_file}", "w") as f:
        f.write(patch)
    o = subprocess.run(
        f"cd {repo_playground} && git apply --whitespace=nowarn {patch_file}",
        shell=True,
        capture_output=True,
    )
    if o.stderr.decode("utf-8"):
        print("stderr> ", o.stderr.decode("utf-8"))
        # TODO: This rarely happen but the patch should be valid, needs to look into it

        with open(f"{repo_playground}/{file_path}", "w") as f:
            f.write(old_content + "\n")

        o = subprocess.run(
            f"cd {repo_playground} && git apply --whitespace=nowarn {patch_file}",
            shell=True,
            capture_output=True,
        )

        if o.stderr.decode("utf-8"):
            print("stderr> ", o.stderr.decode("utf-8"))
            assert False, "shouldn't happen"

    # get git diff
    o = subprocess.run(
        f"cd {repo_playground} && cat {file_path}", shell=True, capture_output=True
    )

    s = o.stdout.decode("utf-8")

    # remove playground
    subprocess.run(f"rm -rf {repo_playground}", shell=True)

    return s


def fake_git_apply_multiple(repo_playground, file_path_contents, patch) -> dict:
    """create a fake git repo to obtain new file contents (multiple)"""

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    # create a fake git repo
    subprocess.run(f"cd {repo_playground} && git init", shell=True)

    # create files
    for file_path, old_content in file_path_contents.items():
        subprocess.run(
            f"mkdir -p {repo_playground}/{os.path.dirname(file_path)}", shell=True
        )

        with open(f"{repo_playground}/{file_path}", "w") as f:
            f.write(old_content)

        # add file to git
        subprocess.run(
            f"cd {repo_playground} && git add {file_path} && git commit -m 'initial commit'",
            shell=True,
        )

    # apply patch file
    patch_file = f"{str(uuid.uuid4())}.patch"
    with open(f"{repo_playground}/{patch_file}", "w") as f:
        f.write(patch)
    o = subprocess.run(
        f"cd {repo_playground} && git apply --whitespace=nowarn {patch_file}",
        shell=True,
        capture_output=True,
    )
    if o.stderr.decode("utf-8"):
        print("stderr> ", o.stderr.decode("utf-8"))
        # TODO: This rarely happen but the patch should be valid, needs to look into it
        for file_path, old_content in file_path_contents.items():
            with open(f"{repo_playground}/{file_path}", "w") as f:
                f.write(old_content + "\n")

        o = subprocess.run(
            f"cd {repo_playground} && git apply --whitespace=nowarn {patch_file}",
            shell=True,
            capture_output=True,
        )

        if o.stderr.decode("utf-8"):
            print("stderr> ", o.stderr.decode("utf-8"))
            assert False, "shouldn't happen"

    new_file_path_contents = {}

    # get git diff
    for file_path, old_content in file_path_contents.items():
        o = subprocess.run(
            f"cd {repo_playground} && cat {file_path}", shell=True, capture_output=True
        )

        s = o.stdout.decode("utf-8")

        new_file_path_contents[file_path] = s

    # remove playground
    subprocess.run(f"rm -rf {repo_playground}", shell=True)

    return new_file_path_contents


def get_functions(tree):
    """Get a set of function and method names from the AST tree."""
    functions = {}

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.parents = []

        def visit(self, node):
            self.parents.append(node)
            super().visit(node)
            self.parents.pop()

        def visit_FunctionDef(self, node):
            if not any(isinstance(parent, ast.ClassDef) for parent in self.parents):
                functions[node.name] = ast.unparse(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            if not any(isinstance(parent, ast.ClassDef) for parent in self.parents):
                functions[node.name] = ast.unparse(node)
            self.generic_visit(node)

    class ClassVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            class_name = node.name
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef) or isinstance(
                    body_item, ast.AsyncFunctionDef
                ):
                    functions[f"{class_name}.{body_item.name}"] = ast.unparse(body_item)
            self.generic_visit(node)

    FunctionVisitor().visit(tree)
    ClassVisitor().visit(tree)
    return functions


def is_just_new_function(code1, code2):
    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)

    functions1 = get_functions(tree1)
    functions2 = get_functions(tree2)

    # The new functions in the second code
    if len(set(list(functions1.keys())) - set(list(functions2.keys()))) > 0:
        # removes functions
        return False

    for func in functions1:
        if functions1[func] != functions2[func]:
            # modifies existing functions
            return False

    if len(set(list(functions2.keys())) - set(list(functions1.keys()))) > 0:
        return True

    # modifying global stuff is okay, because its actually same as functions almost.

    return False


import io
import re
import tokenize


def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += " " * (start_col - last_col)
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = "\n".join(l for l in out.splitlines() if l.strip())
    return out


def normalize_patch(
    instance_id,
    patch: str,
    original_file_content: list,
    new_file_content: list,
    edited_files: list,
) -> str:
    "Remove edits to trailing spaces and comments in the patch."
    if not patch.strip():
        return ""

    normalized_diff = ""

    for o_file_content, n_file_content, edited_file in zip(
        original_file_content, new_file_content, edited_files
    ):
        old_content = o_file_content
        new_content = n_file_content

        # Normalize file contents
        def normalize_code(code):
            try:
                node = ast.parse(code)
                return ast.unparse(node)
            except:
                return code

        old_content = normalize_code(old_content)
        new_content = normalize_code(new_content)

        try:
            remove_docstring_old_content = remove_comments_and_docstrings(old_content)
            ast.parse(remove_docstring_old_content)  # check
            remove_docstring_new_content = remove_comments_and_docstrings(new_content)
            ast.parse(remove_docstring_new_content)  # check
        except:
            # when does this exception happen?
            # when the code has some class or function with empty docstring (thats valid python code)
            # but removing it is not, to be save we just use the original.
            remove_docstring_old_content = old_content
            remove_docstring_new_content = new_content

        diff = fake_git_repo(
            PLAYGROUND_DIR,
            edited_file,
            remove_docstring_old_content,
            remove_docstring_new_content,
        )

        if is_just_new_function(
            remove_docstring_old_content, remove_docstring_new_content
        ):
            # modify the diff to ignore context.
            new_diff = []
            for line in diff.splitlines():
                if line.startswith("-") or line.startswith("+"):
                    new_diff.append(line)
            diff = "\n".join(new_diff)

        normalized_diff += diff

    # Note that the normalized diff may not be applied to the original file.
    return normalized_diff


def extract_python_blocks(text):
    # Regular expression pattern to match ```python\n{text}\n```
    # pattern = r"```python\n(.*?)\n```"
    pattern = r"```(?:python)?\n(.*?)\n```"  

    # Use re.findall to find all matches
    matches = re.findall(pattern, text, re.DOTALL)

    return matches


def extract_code_blocks(text):
    pattern = r"```\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        if "```" in text:
            # handle the case where the code block is not complete
            return [text.split("```", 1)[-1].strip()]
    return matches


def extract_locs_for_files(locs, file_names):
    # TODO: keep the order from this fine-grained FL results.
    results = {fn: [] for fn in file_names}
    current_file_name = None
    for loc in locs:
        for line in loc.splitlines():
            if line.strip().endswith(".py"):
                current_file_name = line.strip()
            elif line.strip() and any(
                line.startswith(w)
                for w in ["line:", "function:", "class:", "variable:"]
            ):
                if current_file_name in results:
                    results[current_file_name].append(line)
                else:
                    pass
    return [["\n".join(results[fn])] for fn in file_names]


def extract_starting_number(subcommand):
    return int(subcommand.split(",")[0].split("start=")[-1])


def extract_ending_number(subcommand):
    return int(subcommand.split(",")[1].split("end=")[-1])


def overlap(subcommand1, subcommand2):
    start1, end1 = extract_starting_number(subcommand1), extract_ending_number(
        subcommand1
    )
    start2, end2 = extract_starting_number(subcommand2), extract_ending_number(
        subcommand2
    )
    return not (end1 < start2 or end2 < start1)


def split_edit_multifile_commands(commands: list[str]) -> dict[str, list[str]]:
    """Split commands based on edited files."""
    file_to_commands = OrderedDict()  # type: ignore
    for command in commands:
        file_name = None
        for subcommand in command.split(">>>>>>> REPLACE")[:-1]:
            subcommand = subcommand.strip()
            if "<<<<<<< SEARCH" in subcommand:
                fn = subcommand.split("<<<<<<< SEARCH")[0].lstrip("#").strip()
                if fn:
                    file_name = fn

            if len(subcommand.split("<<<<<<< SEARCH")) != 2:
                continue
            converted_command = (
                "<<<<<<< SEARCH"
                + subcommand.split("<<<<<<< SEARCH")[1]
                + "\n"
                + ">>>>>>> REPLACE"
            )
            # deduplicate
            if file_name is not None and (
                file_name not in file_to_commands
                or converted_command not in file_to_commands[file_name]
            ):
                file_to_commands.setdefault(file_name, []).append(converted_command)
    return file_to_commands


def parse_diff_edit_commands(commands: list[str], content: str) -> str:
    replaced = False
    # apply the edits from the end of file to the beginning of file
    # this is to make sure context is correct
    # since we want to replace the original context, let's first check for all edits.
    can_apply = []
    for subcommand in commands:
        if not subcommand.startswith("<<<<<<< SEARCH") and subcommand.endswith(
            ">>>>>>> REPLACE"
        ):
            continue

        subcommand = "\n".join(subcommand.splitlines()[1:-1])
        if len(subcommand.split("\n=======\n")) != 2:
            continue

        original, replace = subcommand.split("\n=======\n")

        if original in content:
            can_apply.append(subcommand)

    # apply edits backwards
    # for subcommand in can_apply[::-1]:
    # NOTE(yuxiang): 02/16, not needed; just apply them forwards
    for subcommand in can_apply:
        original, replace = subcommand.split("\n=======\n")
        content = content.replace(original, replace)
        replaced = True

    if not replaced:
        print("not replaced")

    return content

def get_all_filenames(diff: str) -> dict:
    """从 git diff 输出中提取所有变更的文件名
    
    Args:
        diff: git diff 命令的输出文本
        
    Returns:
        dict: 包含添加、修改和删除的文件列表
    """
    modified_files = re.findall(r"diff --git a/(.*?) b/", diff)

    
    added_files = re.findall(r"--- /dev/null\n\+\+\+ b/(.*?)\n@@", diff)
    added_files2 = re.findall(r"diff --git a/(.*?) b/.*?\nnew file mode", diff)  
    added_files = list(set(added_files) | set(added_files2))

    removed_files = re.findall(r"--- a/(.*?)\n\+\+\+ /dev/null\n@@",diff)
    deleted_files = re.findall(r"diff --git a/(.*?) b/.*?\ndeleted file mode", diff)  
    removed_files = list(set(removed_files) | set(deleted_files))

    modified_files = list(set(modified_files) - set(added_files)-set(removed_files))

    return {
        "added": added_files if added_files else [],
        "modified": modified_files if modified_files else [],
        "removed": removed_files if removed_files else [],
    }

def _filter_test_content(origin_content_str: str, class_function_specs_list: list[str]) -> str:
    target_methods_by_class = {}
    for spec in class_function_specs_list:
        parts = spec.split('::', 1)
        if len(parts) == 2:
            scope_key, func_name = parts
            if scope_key not in target_methods_by_class:
                target_methods_by_class[scope_key] = set()
            target_methods_by_class[scope_key].add(func_name)

    try:
        module_node = ast.parse(origin_content_str)
    except SyntaxError:
        print(f"Warning: Could not parse content due to SyntaxError, returning original. Content snippet: {origin_content_str[:100]}...")
        return origin_content_str

    new_module_body = []
    
    # Get the set of targeted top-level functions, if any (used for functions starting with "test")
    targeted_top_level_test_functions = target_methods_by_class.get("", set())

    for node in module_node.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            new_module_body.append(node)  # Always preserve imports
        elif isinstance(node, ast.ClassDef):
            class_name = node.name

            if not class_name.startswith("Test"):
                new_module_body.append(node)  # Keep non-"Test" classes entirely
                continue

            # --- Handle classes starting with "Test" ---
            prospective_class_body = []
            
            # 1. Process and potentially retain docstring from the original class body
            first_stmt_is_docstring_node = None
            if node.body and isinstance(node.body[0], ast.Expr) and \
               isinstance(node.body[0].value, ast.Constant) and \
               isinstance(node.body[0].value.value, str):
                first_stmt_is_docstring_node = node.body[0]
                prospective_class_body.append(first_stmt_is_docstring_node)

            # 2. Iterate through original class members to decide what to keep
            for item_in_class_body in node.body:
                if item_in_class_body == first_stmt_is_docstring_node: # Skip already processed docstring
                    continue

                if isinstance(item_in_class_body, ast.FunctionDef): # It's a method
                    method_node = item_in_class_body
                    method_name = method_node.name

                    if not method_name.startswith("test"):
                        # Keep methods not starting with "test" (e.g., setUp, tearDown, helpers)
                        prospective_class_body.append(method_node)
                    else: # Method name starts with "test"
                        # Keep only if explicitly targeted in class_function_specs_list
                        is_explicitly_targeted = (class_name in target_methods_by_class and
                                                  method_name in target_methods_by_class.get(class_name, set()))
                        if is_explicitly_targeted:
                            prospective_class_body.append(method_node)
                else: 
                    # Keep other class-level statements (class variables, Pass, etc.)
                    prospective_class_body.append(item_in_class_body)
            
            # 3. Decide if the "Test" class itself (with its filtered body) should be kept
            if prospective_class_body: # If the class still has any content after filtering
                node.body = prospective_class_body # Update the class node with the filtered body
                new_module_body.append(node)       # Add the modified class to the module
            # Else, the "Test" class is dropped if it becomes empty after filtering

        elif isinstance(node, ast.FunctionDef): # This is a top-level function
            func_name = node.name
            if not func_name.startswith("test"):
                new_module_body.append(node) # Keep top-level functions not starting with "test"
            else: # Top-level function name starts with "test"
                # Keep only if explicitly targeted in class_function_specs_list (under "" scope_key)
                if func_name in targeted_top_level_test_functions:
                    new_module_body.append(node)
            
        else:
            # Preserve other top-level statements (module docstrings, assignments, Pass, etc.)
            new_module_body.append(node)

    module_node.body = new_module_body
    
    try:
        return ast.unparse(module_node)
    except AttributeError:
        print("Warning: ast.unparse is not available (requires Python 3.9+). "
              "The code modification feature will return original content.")
        return origin_content_str
    except Exception as e:
        # Catch any other potential errors during unparsing
        print(f"Error during ast.unparse: {e}. Returning original content.")
        return origin_content_str

# HELPER FUNCTION to identify 'if __name__ == "__main__":' blocks
def _is_main_if_block_node(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    
    test_expr = node.test
    if not isinstance(test_expr, ast.Compare):
        return False
    
    # Check left operand: ast.Name(id='__name__')
    if not (isinstance(test_expr.left, ast.Name) and test_expr.left.id == '__name__'):
        return False
        
    # Check operator: ast.Eq()
    if not (len(test_expr.ops) == 1 and isinstance(test_expr.ops[0], ast.Eq)):
        return False
        
    # Check right operand: ast.Constant(value='__main__')
    # Assumes Python 3.8+ where strings in AST are ast.Constant
    if not (len(test_expr.comparators) == 1):
        return False
    
    comp = test_expr.comparators[0]
    if not (isinstance(comp, ast.Constant) and comp.value == '__main__'):
        return False
        
    return True

def _generate_combined_test_script(good_tests: dict) -> str:
    unique_import_strs = set()
    final_imports_ast = []
    final_classes_ast = []
    final_other_top_level_ast = []
    
    parse_error_occurred = False
    has_content = False

    # Check if ast.unparse is available for import deduplication, otherwise skip it.
    can_unparse_for_dedup = hasattr(ast, 'unparse')

    for test_file_info in good_tests.values():
        modified_content_str = test_file_info['content']
        if not modified_content_str.strip():
            continue # Skip empty content

        has_content = True
        try:
            module_node = ast.parse(modified_content_str)
            for node in module_node.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if can_unparse_for_dedup:
                        try:
                            import_str = ast.unparse(node)
                            if import_str not in unique_import_strs:
                                unique_import_strs.add(import_str)
                                final_imports_ast.append(node)
                        except Exception: # Fallback if unparsing this node fails
                            final_imports_ast.append(node) # Add it anyway, may lead to duplicates
                    else:
                        # Without unparse, cannot reliably deduplicate, so just add
                        final_imports_ast.append(node)
                elif isinstance(node, ast.ClassDef):
                    final_classes_ast.append(node)
                elif _is_main_if_block_node(node): # Check and skip if it's a main if block
                    continue
                else:
                    final_other_top_level_ast.append(node)
        except SyntaxError as e:
            print(f"Warning: SyntaxError parsing a modified test content piece for combined script, skipping this piece. Error: {e}. Content snippet: {modified_content_str[:100]}...")
            parse_error_occurred = True
        except Exception as e: # Catch other potential parsing errors
            print(f"Warning: Error parsing a modified test content piece for combined script, skipping this piece. Error: {e}. Content snippet: {modified_content_str[:100]}...")
            parse_error_occurred = True

    if not has_content and not final_imports_ast and not final_classes_ast and not final_other_top_level_ast:
        if parse_error_occurred:
            return "# All parts of combined test content failed to parse."
        return "# No runnable combined test content could be generated (no valid parts found)."

    # Ensure 'import unittest' is present
    unittest_imported = False
    if can_unparse_for_dedup:
        for imp_str in unique_import_strs:
            # Basic check, might need to be more robust for aliases or `from unittest import ...`
            if "import unittest" in imp_str or "from unittest" in imp_str:
                unittest_imported = True
                break
    else: # If cannot unparse, assume it might be there or add defensively
        for node in final_imports_ast: # Less reliable check
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "unittest":
                        unittest_imported = True
                        break
            if unittest_imported:
                break
    
    if not unittest_imported:
        final_imports_ast.insert(0, ast.Import(names=[ast.alias(name='unittest', asname=None)]))


    final_body_ast = final_imports_ast + final_other_top_level_ast + final_classes_ast

    if not final_body_ast and not parse_error_occurred: # No actual code statements, only perhaps an auto-added import
        return "# No executable content found in combined tests."
    if not final_body_ast and parse_error_occurred:
        return "# All parts of combined test content failed to parse or resulted in no executable statements."


    # Add `if __name__ == '__main__': unittest.main()`
    main_call = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='unittest', ctx=ast.Load()),
                attr='main',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
    )
    if_main_block = ast.If(
        test=ast.Compare(
            left=ast.Name(id='__name__', ctx=ast.Load()),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value='__main__')] # In Python 3.8+, string constants are ast.Constant
        ),
        body=[main_call],
        orelse=[]
    )
    final_body_ast.append(if_main_block)

    combined_script_content = ""
    try:
        if not hasattr(ast, 'unparse'): # Check specifically for ast.unparse
            raise AttributeError("ast.unparse is not available")
            
        final_module_node = ast.Module(body=final_body_ast, type_ignores=[])
        combined_script_content = ast.unparse(final_module_node)
        if parse_error_occurred:
            combined_script_content = f"# Warning: Some test parts were skipped due to parsing errors.\n{combined_script_content}"
    
    except AttributeError: # Fallback for ast.unparse not available (Python < 3.9)
        print("Warning: ast.unparse is not available (requires Python 3.9+). "
              "Combined script will use fallback string concatenation.")
        all_contents_list = [test_info['content'] for test_info in good_tests.values() if test_info['content'].strip()]
        base_script = "\n\n# --- Next Test File (Fallback Concatenation due to missing ast.unparse) ---\n\n".join(all_contents_list)
        if not unittest_imported and "import unittest" not in base_script : # crude check for fallback
            base_script = "import unittest\n" + base_script
        combined_script_content = base_script + "\n\nif __name__ == '__main__':\n    unittest.main()\n"
        if parse_error_occurred:
            combined_script_content = f"# Warning: Some test parts were skipped during parsing (see logs).\n{combined_script_content}"

    except Exception as e:
        print(f"Error during ast.unparse for combined script: {e}. "
               "Using fallback string concatenation.")
        all_contents_list = [test_info['content'] for test_info in good_tests.values() if test_info['content'].strip()]
        base_script = "\n\n# --- Next Test File (Fallback Concatenation due to unparse error) ---\n\n".join(all_contents_list)
        if not unittest_imported and "import unittest" not in base_script: # crude check for fallback
             base_script = "import unittest\n" + base_script
        combined_script_content = base_script + "\n\nif __name__ == '__main__':\n    unittest.main()\n"
        if parse_error_occurred: # Add warning if parse errors happened before this unparse attempt
            combined_script_content = f"# Warning: Some test parts were skipped during parsing (see logs).\n{combined_script_content}"
            
    return combined_script_content


from envs import API_KEY, BASE_URL, MODEL, MAX_TOKENS, SYSTEM_PROMPT
import requests
import time

def get_llm_response(prompt: str, temperature: float = 0.7) -> str:
    """
    Sends a prompt to a large language model and returns the text response.

    Args:
        prompt: The user's question or instruction.
        max_tokens: Maximum tokens for the response (uses MAX_TOKENS from envs if not provided)
        temperature: Temperature for response randomness

    Returns:
        The text response from the model as a string, or None if the request fails.
    """ 
    headers = {
        'Content-Type': 'application/json',
        'X-Api-Key': API_KEY
    }

    # The payload follows the structure required by the API endpoint.
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "temperature": temperature,
        "messages": [
            {'role': 'user', 'content': prompt}
        ]
    }
    
    # 发送请求，加入重试逻辑
    for retry in range(3):
        try:
            # Make the API call
            response = requests.post(
                BASE_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=600  # 10-minute timeout
            )
            
            # print(f"API Response Status: {response.status_code}")
            if response.status_code == 200:
                response_json = response.json()
                # print(response_json)
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    response_text = response_json["choices"][0]["message"]["content"]
                    return response_text
                else:
                    print(f"Unexpected response format: {response_json}")
            
            error_msg = f"API request error: {response.status_code}, {response.text}"
            print(f"{error_msg} - Retrying ({retry+1}/3)")
            
            # 检查特定错误码
            try:
                error_json = response.json()
                if "error" in error_json and error_json["error"].get("code") == '-4003':
                    print("Rate limit or quota exceeded, returning None")
                    return None
            except:
                pass
                
            time.sleep(10.0)
        
        except requests.exceptions.RequestException as e:
            print(f"API request exception: {str(e)} - Retrying ({retry+1}/3)")
            time.sleep(30)
        except Exception as e:
            print(f"Unexpected error: {str(e)} - Retrying ({retry+1}/3)")
            time.sleep(30)
    
    print("All retry attempts failed")
    return None


def get_deepseek_response(prompt: str, temperature: float = 0.7) -> str:
    """
    Sends a prompt to the DeepSeek model and returns the text response.

    Args:
        prompt: The user's question or instruction.
        temperature: Temperature for response randomness

    Returns:
        The text response from the model as a string, or None if the request fails.
    """
    headers = {
        'Content-Type': 'application/json'
    }

    # The payload follows the structure required by the DeepSeek API endpoint.
    payload = {
        "model": "DeepSeek-V3-0324",
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "extra": {
            "ray_env": "tiktok.aiic.deepseek_zhangbiao168"
        }
    }

    # The URL for the DeepSeek API
    url = 'https://maas.byteintl.net/service/api/v1/chat/completions'

    # Sending the request with retry logic
    for retry in range(3):
        try:
            # Make the API call
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                # timeout=600  # 10-minute timeout
            )

            if response.status_code == 200:
                response_json = response.json()
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    response_text = response_json["choices"][0]["message"]["content"]
                    return response_text
                else:
                    print(f"Unexpected response format: {response_json}")
            
            error_msg = f"API request error: {response.status_code}, {response.text}"
            print(f"{error_msg} - Retrying ({retry+1}/3)")

            try:
                error_json = response.json()
                if "error" in error_json and error_json["error"].get("code") == '-4003':
                    print("Rate limit or quota exceeded, returning None")
                    return None
            except:
                pass
                
            time.sleep(10.0)
        
        except requests.exceptions.RequestException as e:
            print(f"API request exception: {str(e)} - Retrying ({retry+1}/3)")
            time.sleep(30)
        except Exception as e:
            print(f"Unexpected error: {str(e)} - Retrying ({retry+1}/3)")
            time.sleep(30)
    
    print("All retry attempts failed for DeepSeek")
    return None

# ================== prompt construction ==================
def read_yaml(config='default'):
    yaml_file = f'/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/prompt/{config}.yaml'
    with open(yaml_file, 'r', encoding='utf-8') as yaml_file:
        return yaml.safe_load(yaml_file)

def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """
    计算输入字符串的token数。

    :param text: 输入的字符串
    :param model_name: 使用的模型名称，默认为"gpt-4"
    :return: 字符串的token数
    """
    # 获取指定模型的编码器
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        
        # 将字符串编码为token
        tokens = encoding.encode(text)
        
        # 返回token的数量
        return len(tokens)
    except:
        return 1000000


def construct_three_shot_prompt_with_mutiscript(task_item: dict, repo_path: str, sample_data: list, template_path: str, structure: dict) -> str:
    """
    Args:
        task_item: The current task item containing instance_id, repo info, patches, etc.
        repo_path: Base path to the repository (already processed)
        sample_data: List of successful sample tasks to use for examples
        
    Returns:
        str: The reconstructed prompt, or empty string if reconstruction fails
    """
    try:
        # file_content, _, _ = get_full_file_paths_and_classes_and_functions(
        #     structure
        # )
        # repo_file_contents_dict = {path: lines for path, lines in file_content}
        # logger.info(f"Starting prompt reconstruction for task {task_item.get('instance_id', 'unknown')}")
        
        # Load the template for three-shot mode
        template = read_yaml(template_path)
        if not template or 'prompt_template' not in template:
            logger.error("Template not found or invalid for three_shot_same_test mode")
            return ""
        
        # Get repository information
        repo_name = task_item.get('repo', '').replace('/', '__') + '__' + task_item.get('base_commit', '')[:6]
        source_testbed = os.path.join(repo_path, repo_name)
        
        if not os.path.exists(source_testbed):
            logger.warning(f"Source testbed not found: {source_testbed}")
            return ""
        
        # Extract current task's example information
        example_problem_statement_1 = task_item.get('problem_statement', '')
        
        # Read buggy files directly from the processed repo
        example_buggy_files_1 = ""
        input_files = task_item.get('input_files', [])
        
        for file_path in input_files:
            full_file_path = os.path.join(source_testbed, file_path)
            if os.path.exists(full_file_path):
                
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    example_buggy_files_1 += f"**File Path:** {file_path}\n```python\n{file_content}\n```\n\n"
        
        if not example_buggy_files_1:
            logger.warning("No buggy files content found for current task")
            return ""
        
        # Get two additional examples from sample_data
        if len(sample_data) < 2:
            logger.warning("Not enough sample data for three-shot prompt")
            return ""
        
        # Randomly sample examples
        available_samples = [s for s in sample_data if s.get('instance_id') != task_item.get('instance_id')]
        if len(available_samples) < 2:
            logger.warning("Not enough available samples for examples")
            return ""
        
        random_samples = random.sample(available_samples, min(5, len(available_samples)))
        candidate_examples = []
        
        for sample_item in random_samples:
            try:
                sample_problem_statement = sample_item.get('problem_statement', '')
                sample_repo_name = sample_item.get('repo', '').replace('/', '__') + '__' + sample_item.get('base_commit', '')[:6]
                sample_source_testbed = os.path.join(repo_path, sample_repo_name)
                
                if not os.path.exists(sample_source_testbed):
                    continue
                
                # Read sample buggy files directly
                sample_example_buggy_files = ""
                sample_input_files = sample_item.get('input_files', [])
                
                for file_path in sample_input_files:
                    sample_file_full_path = os.path.join(sample_source_testbed, file_path)
                    if os.path.exists(sample_file_full_path):
                        with open(sample_file_full_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                            sample_example_buggy_files += f"**File Path:** {file_path}\n```python\n{file_content}\n```\n\n"
                
                if sample_example_buggy_files:
                    candidate_examples.append({
                        'problem_statement': sample_problem_statement,
                        'buggy_files': sample_example_buggy_files,
                        'buggy_files_length': len(sample_example_buggy_files)
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing sample example: {e}")
                continue
        
        # Select shortest two examples
        if len(candidate_examples) < 2:
            logger.warning("Not enough candidate examples generated")
            return ""
        
        candidate_examples.sort(key=lambda x: x['buggy_files_length'])
        selected_examples = candidate_examples[:2]
        
        # Get original code (clean version before bug)
        
        
        # Read the original files from the repository (assuming they are the clean versions)
        
        main_script = task_item['main_script']
        main_script_path = os.path.join(source_testbed, main_script)
        if os.path.exists(main_script_path):
            with open(main_script_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                main_script_code = f"File Name: {main_script}\n\nFile Content:\n ```python\n{file_content}\n```\n"



        dependencies_script_code = ''
        dependencies_script = task_item['main_script_metadata']['dependencies']
        for file_path in dependencies_script:
            full_file_path = os.path.join(source_testbed, file_path)
            if os.path.exists(full_file_path):
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    dependencies_script_code += f"File Name: {file_path}\n\nFile Content:\n ```python\n{file_content}\n```\n"
        
        # Build final prompt using template
        prompt = template['prompt_template'].format(
            main_script_code=main_script_code,
            dependencies_script_code=dependencies_script_code,
            example_problem_statement_1=example_problem_statement_1,
            example_buggy_files_1=example_buggy_files_1,
            example_problem_statement_2=selected_examples[0]['problem_statement'],
            example_buggy_files_2=selected_examples[0]['buggy_files'],
            example_problem_statement_3=selected_examples[1]['problem_statement'],
            example_buggy_files_3=selected_examples[1]['buggy_files']
        )
        
        # Check token limit
        try:
            if count_tokens(prompt) >= 100000:
                logger.warning("Prompt exceeds token limit")
                return ""
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return ""
        
        # logger.info(f"Successfully reconstructed prompt for task {task_item.get('instance_id', 'unknown')}")
        return prompt
        
    except Exception as e:
        logger.error(f"Error in prompt reconstruction: {e}")
        return ""

    
def construct_three_shot_prompt(task_item: dict, repo_path: str, sample_data: list, template_path: str, structure: dict) -> str:
    """
    Reconstructs a three-shot prompt based on task items, repository paths, and sample data.
    This mirrors the logic used in data_loader.py for the 'three_shot_same_test' mode.
    
    Args:
        task_item: The current task item containing instance_id, repo info, patches, etc.
        repo_path: Base path to the repository (already processed)
        sample_data: List of successful sample tasks to use for examples
        
    Returns:
        str: The reconstructed prompt, or empty string if reconstruction fails
    """
    try:
        # file_content, _, _ = get_full_file_paths_and_classes_and_functions(
        #     structure
        # )
        # repo_file_contents_dict = {path: lines for path, lines in file_content}
        # logger.info(f"Starting prompt reconstruction for task {task_item.get('instance_id', 'unknown')}")
        
        # Load the template for three-shot mode
        template = read_yaml(template_path)
        if not template or 'prompt_template' not in template:
            logger.error("Template not found or invalid for three_shot_same_test mode")
            return ""
        
        # Get repository information
        repo_name = task_item.get('repo', '').replace('/', '__') + '__' + task_item.get('base_commit', '')[:6]
        source_testbed = os.path.join(repo_path, repo_name)
        
        if not os.path.exists(source_testbed):
            logger.warning(f"Source testbed not found: {source_testbed}")
            return ""
        
        # Extract current task's example information
        example_problem_statement_1 = task_item.get('problem_statement', '')
        
        # Read buggy files directly from the processed repo
        example_buggy_files_1 = ""
        input_files = task_item.get('input_files', [])
        noise_files = task_item.get('noise_files', [])
        
        for file_path in input_files + noise_files:
            full_file_path = os.path.join(source_testbed, file_path)
            if os.path.exists(full_file_path):
                
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    example_buggy_files_1 += f"**File Path:** {file_path}\n```python\n{file_content}\n```\n\n"
        
        if not example_buggy_files_1:
            logger.warning("No buggy files content found for current task")
            return ""
        
        # Get two additional examples from sample_data
        if len(sample_data) < 2:
            logger.warning("Not enough sample data for three-shot prompt")
            return ""
        
        # Randomly sample examples
        available_samples = [s for s in sample_data if s.get('instance_id') != task_item.get('instance_id')]
        if len(available_samples) < 2:
            logger.warning("Not enough available samples for examples")
            return ""
        
        random_samples = random.sample(available_samples, min(5, len(available_samples)))
        candidate_examples = []
        
        for sample_item in random_samples:
            try:
                sample_problem_statement = sample_item.get('problem_statement', '')
                sample_repo_name = sample_item.get('repo', '').replace('/', '__') + '__' + sample_item.get('base_commit', '')[:6]
                sample_source_testbed = os.path.join(repo_path, sample_repo_name)
                
                if not os.path.exists(sample_source_testbed):
                    continue
                
                # Read sample buggy files directly
                sample_example_buggy_files = ""
                sample_input_files = sample_item.get('input_files', [])
                sample_noise_files = sample_item.get('noise_files', [])
                
                for file_path in sample_input_files + sample_noise_files:
                    sample_file_full_path = os.path.join(sample_source_testbed, file_path)
                    if os.path.exists(sample_file_full_path):
                        with open(sample_file_full_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                            sample_example_buggy_files += f"**File Path:** {file_path}\n```python\n{file_content}\n```\n\n"
                
                if sample_example_buggy_files:
                    candidate_examples.append({
                        'problem_statement': sample_problem_statement,
                        'buggy_files': sample_example_buggy_files,
                        'buggy_files_length': len(sample_example_buggy_files)
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing sample example: {e}")
                continue
        
        # Select shortest two examples
        if len(candidate_examples) < 2:
            logger.warning("Not enough candidate examples generated")
            return ""
        
        candidate_examples.sort(key=lambda x: x['buggy_files_length'])
        selected_examples = candidate_examples[:2]
        
        # Get original code (clean version before bug)
        original_code = ''
        
        # Read the original files from the repository (assuming they are the clean versions)
        for file_path in input_files + noise_files:
            full_file_path = os.path.join(source_testbed, file_path)
            if os.path.exists(full_file_path):
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    original_code += f"File Name: {file_path}\n\nFile Content:\n ```python\n{file_content}\n```\n"
        
        # Build final prompt using template
        prompt = template['prompt_template'].format(
            original_code=original_code,
            example_problem_statement_1=example_problem_statement_1,
            example_buggy_files_1=example_buggy_files_1,
            example_problem_statement_2=selected_examples[0]['problem_statement'],
            example_buggy_files_2=selected_examples[0]['buggy_files'],
            example_problem_statement_3=selected_examples[1]['problem_statement'],
            example_buggy_files_3=selected_examples[1]['buggy_files']
        )
        
        # Check token limit
        try:
            if count_tokens(prompt) >= 100000:
                logger.warning("Prompt exceeds token limit")
                return ""
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return ""
        
        # logger.info(f"Successfully reconstructed prompt for task {task_item.get('instance_id', 'unknown')}")
        return prompt
        
    except Exception as e:
        logger.error(f"Error in prompt reconstruction: {e}")
        return ""


def origin_prompt(task_item: dict, repo_path: str, sample_data: list) -> str:
    return task_item['prompt']

# ================== retry unittest prompt==================
def construct_unittest_prompt_with_mutiscript(task: dict) -> str:
    repo_path = DEFAULT_PATH
    instance_id = task.get('instance_id', 'unknown')
    logger.info(f"Retrying unittest generation for task {instance_id}")
    
    # 构建重试prompt
    repo_name = task.get('repo', '').replace('/', '__') + '__' + task.get('base_commit', '')[:6]
    source_testbed = os.path.join(repo_path, repo_name)
    
    if not os.path.exists(source_testbed):
        logger.warning(f"Source testbed not found: {source_testbed}")
        return None
    
    main_script = task['main_script']
    main_script_path = os.path.join(source_testbed, main_script)
    if os.path.exists(main_script_path):
        with open(main_script_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            main_script_code = f"File Name: {main_script}\n\nFile Content:\n ```python\n{file_content}\n```\n"

    dependencies_script_code = ''
    dependencies_script = task['main_script_metadata']['dependencies']
    for file_path in dependencies_script:
        full_file_path = os.path.join(source_testbed, file_path)
        if os.path.exists(full_file_path):
            with open(full_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                dependencies_script_code += f"File Name: {file_path}\n\nFile Content:\n ```python\n{file_content}\n```\n"
    # 构建重试prompt
    retry_template = read_yaml('unittest_retry_mutiscript')
    if not retry_template or 'prompt_template' not in retry_template:
        logger.error("unittest_retry template not found")
        return None
    
    problem_statement = task.get('gpt_problem_statement', '')
    if not problem_statement:
        logger.warning(f"No problem statement found for task {instance_id}")
        return None
    
    retry_prompt = retry_template['prompt_template'].format(
        main_script_code=main_script_code,
        dependencies_script_code=dependencies_script_code,
        problem_statement=problem_statement
    )
    return retry_prompt


def construct_unittest_prompt(task: dict) -> str:
    repo_path = DEFAULT_PATH
    instance_id = task.get('instance_id', 'unknown')
    logger.info(f"Retrying unittest generation for task {instance_id}")
    
    # 构建重试prompt
    repo_name = task.get('repo', '').replace('/', '__') + '__' + task.get('base_commit', '')[:6]
    source_testbed = os.path.join(repo_path, repo_name)
    
    if not os.path.exists(source_testbed):
        logger.warning(f"Source testbed not found: {source_testbed}")
        return None
    
    # 读取原始代码
    original_code = ''
    input_files = task.get('input_files', [])
    noise_files = task.get('noise_files', [])
    
    for file_path in input_files + noise_files:
        full_file_path = os.path.join(source_testbed, file_path)
        if os.path.exists(full_file_path):
            with open(full_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                original_code += f"File Name: {file_path}\n\nFile Content:\n ```python\n{file_content}\n```\n"
    
    if not original_code:
        logger.warning(f"No original code found for task {instance_id}")
        return None
    
    # 构建重试prompt
    retry_template = read_yaml('unittest_retry')
    if not retry_template or 'prompt_template' not in retry_template:
        logger.error("unittest_retry template not found")
        return None
    
    problem_statement = task.get('gpt_problem_statement', '')
    if not problem_statement:
        logger.warning(f"No problem statement found for task {instance_id}")
        return None
    
    retry_prompt = retry_template['prompt_template'].format(
        original_code=original_code,
        problem_statement=problem_statement
    )
    return retry_prompt

# retry buggy prompt
def construct_buggy_prompt(task: dict) -> str:
    repo_path = DEFAULT_PATH
    instance_id = task.get('instance_id', 'unknown')
    logger.info(f"Retrying buggy code generation for task {instance_id}")
    # Load the buggy retry prompt
    buggy_retry_prompt = read_yaml('buggy_retry')['prompt_template']
    
    # Extract necessary information from the task
    problem_statement = task.get('gpt_problem_statement', '')
    unittest_code = task.get('unittest_file_code', '')
    repo_name = task.get('repo', '').replace('/', '__') + '__' + task.get('base_commit', '')[:6]
    source_testbed = os.path.join(repo_path, repo_name)
    
    if not os.path.exists(source_testbed):
        logger.warning(f"Source testbed not found: {source_testbed}")
        return None
    original_code = ''
    input_files = task.get('input_files', [])
    noise_files = task.get('noise_files', [])
    
    for file_path in input_files + noise_files:
        full_file_path = os.path.join(source_testbed, file_path)
        if os.path.exists(full_file_path):
            with open(full_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                original_code += f"File Name: {file_path}\n\nFile Content:\n ```python\n{file_content}\n```\n"
    
    # Format the prompt with task-specific information
    formatted_prompt = buggy_retry_prompt.format(
        problem_statement=problem_statement,
        unittest_code=unittest_code,
        original_code=original_code
    )
    return formatted_prompt

# ================== script ranker prompt ==================
def script_ranker_prompt(structure: dict, truncate: int) -> str:
    """
    从repository结构中提取所有.py文件（排除test文件），并移除函数内容，
    然后将结果格式化为字符串作为prompt输入。
    
    Args:
        structure: 表示repository结构的字典
        
    Returns:
        str: 格式化的prompt字符串
        list: 所有有效py文件路径
    """
    file_content_dic = extract_python_files_without_tests(structure)
    valid_files = list(file_content_dic.keys())
    code_content = ""
    # 将字典格式化为字符串
    for file_path, content in file_content_dic.items():
        code_content += f"===FILE_START===\n"
        code_content += f"FILE_PATH: {file_path}\n"
        code_content += f"===CODE_START===\n"
        code_content += f"```python\n{content[:truncate]}\n```\n"
        code_content += f"===CODE_END===\n"
        code_content += f"===FILE_END===\n\n"
    
    template = read_yaml('script_ranker')
    ranker_prompt = template['prompt_template'].format(
        repository_structure=code_content,
    )

    return ranker_prompt, valid_files

def extract_python_files_without_tests(structure, current_path="") -> dict:
    """
    递归遍历repository结构，提取所有.py文件（排除包含test的文件），
    并移除函数和类方法的内容。
    
    Args:
        structure: 表示repository结构的字典
        current_path: 当前路径（用于递归）
        
    Returns:
        dict: 键为文件路径，值为处理后的文件内容
    """
    result = {}
    
    for name, content in structure.items():
        if isinstance(content, dict):
            # 检查是否为文件节点（包含text、classes、functions键）
            is_file_node = (
                "text" in content and 
                isinstance(content.get("text"), list) and
                ("classes" in content or "functions" in content)
            )
            
            if is_file_node:
                # 这是一个Python文件节点
                file_path = f"{current_path}/{name}" if current_path else name
                
                # 检查是否为.py文件且路径不包含test
                if name.endswith('.py') and 'test' not in file_path:
                    try:
                        # 获取原始文件内容
                        file_lines = content.get("text", [])
                        file_content = '\n'.join(file_lines)
                        
                        # 使用AST解析并移除函数内容
                        processed_content = remove_function_bodies(file_content)
                        result[file_path] = processed_content
                        
                    except Exception as e:
                        # 如果解析失败，使用原始内容但移除空行
                        file_lines = content.get("text", [])
                        file_content = '\n'.join([line for line in file_lines if line.strip()])
                        result[file_path] = file_content
                        
            else:
                # 这是一个目录，继续递归
                next_path = f"{current_path}/{name}" if current_path else name
                sub_result = extract_python_files_without_tests(content, next_path)
                result.update(sub_result)
    
    return result

def remove_function_bodies(file_content: str) -> str:
    """
    使用AST解析Python文件内容，移除所有函数和类方法的内容，用占位符代替。
    但保留main()函数和if __name__ == "__main__":块的内容。
    同时移除所有三引号注释（包括\"\"\"和\'\'\'）。
    
    Args:
        file_content: 原始Python文件内容
        
    Returns:
        str: 处理后的文件内容，普通函数体被替换为"# ... function body ..."
    """
    
    def remove_docstrings_and_comments(content: str) -> str:
        """移除所有三引号注释和文档字符串"""
        import re
        
        # 移除三引号注释（非贪婪匹配）
        # 匹配"""..."""和'''...'''
        content = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', content, flags=re.DOTALL)
        
        # 移除单行注释
        lines = content.splitlines()
        filtered_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith('#'):
                # 移除行内的#注释，但要小心处理字符串中的#
                # 使用正则表达式匹配不在字符串中的#
                line_without_comment = re.sub(r'#.*$', '', line)
                if line_without_comment.strip():
                    filtered_lines.append(line_without_comment.rstrip())
                else:
                    filtered_lines.append('')
            else:
                filtered_lines.append('')
        
        return '\n'.join(filtered_lines)
    
    # 首先移除注释和文档字符串
    file_content = remove_docstrings_and_comments(file_content)
    
    try:
        import ast
        tree = ast.parse(file_content)
        lines = file_content.splitlines()
        
        # 收集所有需要替换的函数/方法范围
        replace_ranges = []
        
        # 检查是否是main函数或__main__块
        def should_preserve_function(node):
            """检查是否应该保留函数内容"""
            if isinstance(node, ast.FunctionDef):
                return node.name == 'main'
            elif isinstance(node, ast.AsyncFunctionDef):
                return node.name == 'main'
            return False
        
        # 检查是否是if __name__ == "__main__"块
        def find_main_guard_statements(tree):
            """找到if __name__ == "__main__":语句的范围"""
            main_guard_ranges = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    # 检查条件是否为 __name__ == "__main__"
                    if (isinstance(node.test, ast.Compare) and
                        isinstance(node.test.left, ast.Name) and
                        node.test.left.id == '__name__' and
                        len(node.test.ops) == 1 and
                        isinstance(node.test.ops[0], ast.Eq) and
                        len(node.test.comparators) == 1 and
                        isinstance(node.test.comparators[0], ast.Constant) and
                        node.test.comparators[0].value == '__main__'):
                        
                        # 找到if语句的起始和结束行
                        start_line = node.lineno - 1
                        body_end = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                        
                        # 保留整个if块
                        main_guard_ranges.append((start_line, body_end))
            
            return main_guard_ranges
        
        # 找到所有需要保留的main guard范围
        preserved_main_guards = find_main_guard_statements(tree)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 检查是否应该保留这个函数
                if should_preserve_function(node):
                    continue  # 跳过main函数，不添加到替换范围
                
                # 找到函数定义的起始和结束行
                start_line = node.lineno - 1
                
                # 找到函数体的起始位置
                func_def_line = lines[start_line]
                colon_pos = func_def_line.rfind(':')
                
                if colon_pos != -1:
                    body_start = start_line + 1
                else:
                    for i in range(start_line, len(lines)):
                        if ':' in lines[i]:
                            body_start = i + 1
                            break
                    else:
                        body_start = start_line + 1
                
                body_end = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                
                # 检查是否与main guard重叠
                should_skip = False
                for guard_start, guard_end in preserved_main_guards:
                    if start_line >= guard_start and body_end <= guard_end:
                        should_skip = True
                        break
                
                if not should_skip:
                    replace_ranges.append((body_start, body_end))
            
            elif isinstance(node, ast.ClassDef):
                # 处理类中的方法
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # 检查是否应该保留这个方法
                        if should_preserve_function(method):
                            continue
                        
                        start_line = method.lineno - 1
                        
                        # 找到方法体的起始位置
                        method_def_line = lines[start_line]
                        colon_pos = method_def_line.rfind(':')
                        
                        if colon_pos != -1:
                            body_start = start_line + 1
                        else:
                            for i in range(start_line, len(lines)):
                                if ':' in lines[i]:
                                    body_start = i + 1
                                    break
                            else:
                                body_start = start_line + 1
                        
                        body_end = method.end_lineno if hasattr(method, 'end_lineno') else len(lines)
                        
                        # 检查是否与main guard重叠
                        should_skip = False
                        for guard_start, guard_end in preserved_main_guards:
                            if start_line >= guard_start and body_end <= guard_end:
                                should_skip = True
                                break
                        
                        if not should_skip:
                            replace_ranges.append((body_start, body_end))
        
        # 按行号降序排序，避免替换时影响后续索引
        replace_ranges.sort(key=lambda x: x[0], reverse=True)
        
        # 执行替换
        for start, end in replace_ranges:
            # 计算缩进
            if start < len(lines):
                indent = len(lines[start]) - len(lines[start].lstrip())
                placeholder = ' ' * indent + '# ... function body ...'
            else:
                placeholder = '# ... function body ...'
            
            # 替换函数体
            lines[start:end] = [placeholder]
        
        # 过滤掉空行
        filtered_lines = [line for line in lines if line.strip()]
        return '\n'.join(filtered_lines)
        
    except Exception:
        # 如果AST解析失败，使用正则表达式方法
        import re
        
        # 再次确保移除注释
        file_content = remove_docstrings_and_comments(file_content)
        lines = file_content.splitlines()
        result_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            if not line.strip():
                i += 1
                continue
            
            # 检查是否是if __name__ == "__main__":块
            if line.strip() == 'if __name__ == "__main__":':
                # 保留整个__main__块
                result_lines.append(line)
                i += 1
                
                # 找到整个块的范围
                indent = len(line) - len(line.lstrip())
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip() and not next_line.startswith(' ' * indent):
                        break
                    result_lines.append(next_line)
                    i += 1
                continue
            
            # 检查是否是main函数定义
            main_func_pattern = r'^\s*(def|async def)\s+main\s*\([^)]*\)\s*:.*'
            if re.match(main_func_pattern, line):
                # 保留main函数
                result_lines.append(line)
                i += 1
                
                # 找到main函数的范围
                func_indent = len(line) - len(line.lstrip())
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= func_indent and not next_line.startswith(' ' * (func_indent + 1)):
                        break
                    result_lines.append(next_line)
                    i += 1
                continue
            
            # 检查普通函数定义
            func_pattern = r'^\s*(def|async def)\s+\w+\s*\([^)]*\)\s*:.*'
            if re.match(func_pattern, line) and not re.match(main_func_pattern, line):
                result_lines.append(line)
                i += 1
                
                # 跳过函数体，添加占位符
                func_indent = len(line) - len(line.lstrip())
                placeholder = ' ' * (func_indent + 4) + '# ... function body ...'
                result_lines.append(placeholder)
                
                # 跳过到函数结束
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= func_indent and not next_line.startswith(' ' * (func_indent + 1)):
                        break
                    i += 1
                continue
            
            # 普通行，直接添加
            result_lines.append(line)
            i += 1
        
        # 过滤掉空行
        filtered_lines = [line for line in result_lines if line.strip()]
        return '\n'.join(filtered_lines)
# ================== script ranker prompt done ==================

def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

    return class_info, function_names, file_content.splitlines()



def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}

    for root, _, files in os.walk(directory_path):
        relative_root = os.path.relpath(root, directory_path)
        # if relative_root == ".":
        #     relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure



def get_project_structure_from_scratch(
    repo_name, commit_id, instance_id, repo_playground
):

    # Generate a temperary folder and add uuid to avoid collision
    # repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    # assert not os.path.exists(repo_playground), f"{repo_playground} already exists"
    # repo_playground = os.path.join(repo_playground, )
    # repo_base_name = repo_name.split("/")[-1]
    structure = create_structure(repo_playground)
    
    # Move everything from structure['.'] to the root level if '.' exists
    if '.' in structure:
        dot_content = structure['.']
        structure.pop('.')  # Remove the '.' key
        structure.update(dot_content)  # Move all content to root level
    
    d = {
        "repo": repo_name,
        "base_commit": commit_id,
        "structure": structure,
        "instance_id": instance_id,
    }
    return d




if __name__ == "__main__":
    inputs = 'test'
    response = get_llm_response(inputs)
    print(response)