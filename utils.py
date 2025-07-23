# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import ast
import json
import os
import random
import re
import subprocess
import uuid
from collections import OrderedDict

from envs import PLAYGROUND_DIR, PROJECT_FILE_LOC


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


if __name__ == "__main__":
    inputs = 'Given three examples of bugs introduced to Python code and a target Python code snippet, generate a new bug that follows similar patterns of realistic programming errors along with a comprehensive unittest file that can detect this bug.\n\n## Bug Example 1:\n**Problem Statement:** [BUG]description for query paramters can not show in swagger ui\nHi, when I add a description for a schema used in query, it can not show in swagger ui but can show in Redoc\r\n```py\r\n@HELLO.route(\'/\', methods=[\'GET\'])\r\n@api.validate(query=HelloForm)\r\ndef hello():\r\n    """\r\n    hello 注释\r\n    :return:\r\n    """\r\n   return \'ok\'\r\n\r\nclass HelloForm(BaseModel):\r\n    """\r\n    hello表单\r\n    """\r\n    user: str # 用户名称\r\n    msg: str = Field(description=\'msg test\', example=\'aa\')\r\n    index: int\r\n    data: HelloGetListForm\r\n    list: List[HelloListForm]\r\n```\r\n\r\n![截屏2020-10-12 下午7 54 52](https://user-images.githubusercontent.com/60063723/95743785-de70f480-0cc4-11eb-857b-fffd3d7e9cdd.png)\r\n![截屏2020-10-12 下午7 53 59](https://user-images.githubusercontent.com/60063723/95743805-e5980280-0cc4-11eb-99ae-11e6439bae02.png)\r\n\r\n\r\n\n\n**Modified Files:**\n**File Path:** spectree/utils.py\n```python\nimport re\nimport inspect\nimport logging\n\n# parse HTTP status code to get the code\nHTTP_CODE = re.compile(r\'^HTTP_(?P<code>\\d{3})$\')\n\nlogger = logging.getLogger(__name__)\n\n\ndef parse_comments(func):\n    """\n    parse function comments\n\n    First line of comments will be saved as summary, and the rest\n    will be saved as description.\n    """\n    doc = inspect.getdoc(func)\n    if doc is None:\n        return None, None\n    doc = doc.split(\'\\n\', 1)\n    if len(doc) == 1:\n        return doc[0], None\n    return doc[0], doc[1].strip()\n\n\ndef parse_request(func):\n    """\n    get json spec\n    """\n    data = {}\n    if hasattr(func, \'json\'):\n        data = {\n            \'content\': {\n                \'application/json\': {\n                    \'schema\': {\n                        \'$ref\': f\'#/components/schemas/{func.json}\'\n                    }\n                }\n            }\n        }\n    return data\n\n\ndef parse_params(func, params, models):\n    """\n    get spec for (query, headers, cookies)\n    """\n    if hasattr(func, \'query\'):\n        query = models[func.query]\n        for name, schema in query[\'properties\'].items():\n            params.append({\n                \'name\': name,\n                \'in\': \'query\',\n                \'schema\': schema,\n                \'required\': name in query.get(\'required\', []),\n                \'description\': schema.get(\'description\', \'\'),\n            })\n\n    if hasattr(func, \'headers\'):\n        headers = models[func.headers]\n        for name, schema in headers[\'properties\'].items():\n            params.append({\n                \'name\': name,\n                \'in\': \'header\',\n                \'schema\': schema,\n                \'required\': name in headers.get(\'required\', []),\n                \'description\': schema.get(\'description\', \'\'),\n            })\n\n    if hasattr(func, \'cookies\'):\n        cookies = models[func.cookies]\n        for name, schema in cookies[\'properties\'].items():\n            params.append({\n                \'name\': name,\n                \'in\': \'cookie\',\n                \'schema\': schema,\n                \'required\': name in cookies.get(\'required\', []),\n                \'description\': schema.get(\'description\', \'\'),\n            })\n\n    return params\n\n\ndef parse_resp(func):\n    """\n    get the response spec\n\n    If this function does not have explicit ``resp`` but have other models,\n    a ``422 Validation Error`` will be append to the response spec. Since\n    this may be triggered in the validation step.\n    """\n    responses = {}\n    if hasattr(func, \'resp\'):\n        responses = func.resp.generate_spec()\n\n    if \'422\' not in responses and has_model(func):\n        responses[\'422\'] = {\'description\': \'Validation Error\'}\n\n    return responses\n\n\ndef has_model(func):\n    """\n    return True if this function have ``pydantic.BaseModel``\n    """\n    if any(hasattr(func, x) for x in (\'query\', \'json\', \'headers\')):\n        return True\n\n    if hasattr(func, \'resp\') and func.resp.has_model():\n        return True\n\n    return False\n\n\ndef parse_code(http_code):\n    """\n    get the code of this HTTP status\n\n    :param str http_code: format like ``HTTP_200``\n    """\n    match = HTTP_CODE.match(http_code)\n    if not match:\n        return None\n    return match.group(\'code\')\n\n\ndef parse_name(func):\n    """\n    the func can be\n\n        * undecorated functions\n        * decorated functions\n        * decorated class methods\n    """\n    return func.__name__\n\n\ndef default_before_handler(req, resp, req_validation_error, instance):\n    """\n    default handler called before the endpoint function after the request validation\n\n    :param req: request provided by the web framework\n    :param resp: response generated by SpecTree that will be returned\n        if the validation error is not None\n    :param req_validation_error: request validation error\n    :param instance: class instance if the endpoint function is a class method\n    """\n    if req_validation_error:\n        logger.info(\n            \'422 Validation Error\',\n            extra={\n                \'spectree_model\': req_validation_error.model.__name__,\n                \'spectree_validation\': req_validation_error.errors(),\n            },\n        )\n\n\ndef default_after_handler(req, resp, resp_validation_error, instance):\n    """\n    default handler called after the response validation\n\n    :param req: request provided by the web framework\n    :param resp: response from the endpoint function (if there is no validation error)\n        or response validation error\n    :param resp_validation_error: response validation error\n    :param instance: class instance if the endpoint function is a class method\n    """\n    if resp_validation_error:\n        logger.info(\n            \'500 Response Validation Error\',\n            extra={\n                \'spectree_model\': resp_validation_error.model.__name__,\n                \'spectree_validation\': resp_validation_error.errors(),\n            },\n        )\n\n```\n\n**File Path:** setup.py\n```python\nfrom setuptools import setup, find_packages\nfrom os import path\nfrom io import open\n\n\nhere = path.abspath(path.dirname(__file__))\n\nwith open(path.join(here, \'README.md\'), encoding=\'utf-8\') as f:\n    readme = f.read()\n\nwith open(path.join(here, \'requirements.txt\'), encoding=\'utf-8\') as f:\n    requires = [req.strip() for req in f if req]\n\n\nsetup(\n    name=\'spectree\',\n    version=\'0.3.8\',\n    author=\'Keming Yang\',\n    author_email=\'kemingy94@gmail.com\',\n    description=(\'generate OpenAPI document and validate request&response \'\n                 \'with Python annotations.\'),\n    long_description=readme,\n    long_description_content_type=\'text/markdown\',\n    url=\'https://github.com/0b01001001/spectree\',\n    packages=find_packages(exclude=[\'examples*\', \'tests*\']),\n    package_data={\n    },\n    classifiers=[\n        \'Programming Language :: Python :: 3 :: Only\',\n        \'Programming Language :: Python :: 3.6\',\n        \'Programming Language :: Python :: 3.7\',\n        \'Programming Language :: Python :: 3.8\',\n        \'Operating System :: OS Independent\',\n        \'Topic :: Software Development :: Libraries :: Python Modules\',\n    ],\n    python_requires=\'>=3.6\',\n    install_requires=requires,\n    extras_require={\n        \'flask\': [\'flask\'],\n        \'falcon\': [\'falcon\'],\n        \'starlette\': [\'starlette\', \'requests\'],\n    },\n    zip_safe=False,\n    entry_points={\n        \'console_scripts\': [],\n    },\n)\n\n```\n\n**File Path:** spectree/plugins/starlette_plugin.py\n```python\nimport inspect\nfrom collections import namedtuple\nfrom functools import partial\nfrom json import JSONDecodeError\nfrom json import loads as json_loads\n\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin, Context\nfrom .page import PAGES\n\nMETHODS = {\'get\', \'post\', \'put\', \'patch\', \'delete\'}\nRoute = namedtuple(\'Route\', [\'path\', \'methods\', \'func\'])\n\n\nclass StarlettePlugin(BasePlugin):\n    def __init__(self, spectree):\n        super().__init__(spectree)\n        from starlette.convertors import CONVERTOR_TYPES\n        self.conv2type = {\n            conv: typ for typ, conv in CONVERTOR_TYPES.items()\n        }\n\n    def register_route(self, app):\n        self.app = app\n        from starlette.responses import JSONResponse, HTMLResponse\n\n        self.app.add_route(\n            self.config.spec_url,\n            lambda request: JSONResponse(self.spectree.spec),\n        )\n\n        for ui in PAGES:\n            self.app.add_route(\n                f\'/{self.config.PATH}/{ui}\',\n                lambda request, ui=ui: HTMLResponse(\n                    PAGES[ui].format(self.config.spec_url)\n                ),\n            )\n\n    async def request_validation(self, request, query, json, headers, cookies):\n        request.context = Context(\n            query.parse_obj(request.query_params) if query else None,\n            json.parse_obj(json_loads(await request.body() or \'{}\')) if json else None,\n            headers.parse_obj(request.headers) if headers else None,\n            cookies.parse_obj(request.cookies) if cookies else None,\n        )\n\n    async def validate(self,\n                       func,\n                       query, json, headers, cookies, resp,\n                       before, after,\n                       *args, **kwargs):\n        from starlette.responses import JSONResponse\n\n        # NOTE: If func is a `HTTPEndpoint`, it should have \'.\' in its ``__qualname__``\n        # This is not elegant. But it seems `inspect` doesn\'t work here.\n        instance = args[0] if \'.\' in func.__qualname__ else None\n        request = args[1] if \'.\' in func.__qualname__ else args[0]\n        response = None\n        req_validation_error, resp_validation_error, json_decode_error = None, None, None\n\n        try:\n            await self.request_validation(request, query, json, headers, cookies)\n        except ValidationError as err:\n            req_validation_error = err\n            response = JSONResponse(err.errors(), 422)\n        except JSONDecodeError as err:\n            json_decode_error = err\n            self.logger.info(\n                \'422 Validation Error\',\n                extra={\'spectree_json_decode_error\': str(err)}\n            )\n            response = JSONResponse({\'error_msg\': str(err)}, 422)\n\n        before(request, response, req_validation_error, instance)\n        if req_validation_error or json_decode_error:\n            return response\n\n        if inspect.iscoroutinefunction(func):\n            response = await func(*args, **kwargs)\n        else:\n            response = func(*args, **kwargs)\n\n        if resp:\n            model = resp.find_model(response.status_code)\n            if model:\n                try:\n                    model.validate(json_loads(response.body))\n                except ValidationError as err:\n                    resp_validation_error = err\n                    response = JSONResponse(err.errors(), 500)\n\n        after(request, response, resp_validation_error, instance)\n\n        return response\n\n    def find_routes(self):\n        routes = []\n\n        def parse_route(app, prefix=\'\'):\n            for route in app.routes:\n                if route.path.startswith(f\'/{self.config.PATH}\'):\n                    continue\n\n                func = route.app\n                if isinstance(func, partial):\n                    try:\n                        func = func.__wrapped__\n                    except AttributeError:\n                        pass\n\n                if inspect.isclass(func):\n                    for method in METHODS:\n                        if getattr(func, method, None):\n                            routes.append(Route(\n                                f\'{prefix}{route.path}\',\n                                {method.upper()},\n                                getattr(func, method)\n                            ))\n                elif inspect.isfunction(func):\n                    routes.append(Route(\n                        f\'{prefix}{route.path}\',\n                        route.methods,\n                        route.endpoint))\n                else:\n                    parse_route(route, prefix=f\'{prefix}{route.path}\')\n\n        parse_route(self.app)\n        return routes\n\n    def bypass(self, func, method):\n        if method in [\'HEAD\', \'OPTIONS\']:\n            return True\n        return False\n\n    def parse_func(self, route):\n        for method in route.methods or [\'GET\']:\n            yield method, route.func\n\n    def parse_path(self, route):\n        from starlette.routing import compile_path\n        _, path, variables = compile_path(route.path)\n        parameters = []\n\n        for name, conv in variables.items():\n            schema = None\n            typ = self.conv2type[conv]\n            if typ == \'int\':\n                schema = {\n                    \'type\': \'integer\',\n                    \'format\': \'int32\'\n                }\n            elif typ == \'float\':\n                schema = {\n                    \'type\': \'number\',\n                    \'format\': \'float\',\n                }\n            elif typ == \'path\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'path\',\n                }\n            elif typ == \'str\':\n                schema = {\'type\': \'string\'}\n\n            parameters.append({\n                \'name\': name,\n                \'in\': \'path\',\n                \'required\': True,\n                \'schema\': schema,\n            })\n\n        return path, parameters\n\n```\n\n**File Path:** spectree/plugins/falcon_plugin.py\n```python\nimport inspect\nimport re\nfrom functools import partial\n\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin\nfrom .page import PAGES\n\n\nclass OpenAPI:\n    def __init__(self, spec):\n        self.spec = spec\n\n    def on_get(self, req, resp):\n        resp.media = self.spec\n\n\nclass DocPage:\n    def __init__(self, html, spec_url):\n        self.page = html.format(spec_url)\n\n    def on_get(self, req, resp):\n        resp.content_type = \'text/html\'\n        resp.body = self.page\n\n\nDOC_CLASS = [x.__name__ for x in (DocPage, OpenAPI)]\n\n\nclass FalconPlugin(BasePlugin):\n    def __init__(self, spectree):\n        super().__init__(spectree)\n        from falcon.routing.compiled import _FIELD_PATTERN\n\n        self.FIELD_PATTERN = _FIELD_PATTERN\n        # NOTE from `falcon.routing.compiled.CompiledRouterNode`\n        self.ESCAPE = r\'[\\.\\(\\)\\[\\]\\?\\$\\*\\+\\^\\|]\'\n        self.ESCAPE_TO = r\'\\\\\\g<0>\'\n        self.EXTRACT = r\'{\\2}\'\n        # NOTE this regex is copied from werkzeug.routing._converter_args_re and\n        # modified to support only int args\n        self.INT_ARGS = re.compile(r"""\n            ((?P<name>\\w+)\\s*=\\s*)?\n            (?P<value>\\d+)\\s*\n        """, re.VERBOSE)\n        self.INT_ARGS_NAMES = (\'num_digits\', \'min\', \'max\')\n\n    def register_route(self, app):\n        self.app = app\n        self.app.add_route(\n            self.config.spec_url, OpenAPI(self.spectree.spec)\n        )\n        for ui in PAGES:\n            self.app.add_route(\n                f\'/{self.config.PATH}/{ui}\',\n                DocPage(PAGES[ui], self.config.spec_url),\n            )\n\n    def find_routes(self):\n        routes = []\n\n        def find_node(node):\n            if node.resource and node.resource.__class__.__name__ not in DOC_CLASS:\n                routes.append(node)\n\n            for child in node.children:\n                find_node(child)\n\n        for route in self.app._router._roots:\n            find_node(route)\n\n        return routes\n\n    def parse_func(self, route):\n        return route.method_map.items()\n\n    def parse_path(self, route):\n        subs, parameters = [], []\n        for segment in route.uri_template.strip(\'/\').split(\'/\'):\n            matches = self.FIELD_PATTERN.finditer(segment)\n            if not matches:\n                subs.append(segment)\n                continue\n\n            escaped = re.sub(self.ESCAPE, self.ESCAPE_TO, segment)\n            subs.append(self.FIELD_PATTERN.sub(self.EXTRACT, escaped))\n\n            for field in matches:\n                variable, converter, argstr = [field.group(name) for name in\n                                               (\'fname\', \'cname\', \'argstr\')]\n\n                if converter == \'int\':\n                    if argstr is None:\n                        argstr = \'\'\n\n                    arg_values = [None, None, None]\n                    for index, match in enumerate(self.INT_ARGS.finditer(argstr)):\n                        name, value = match.group(\'name\'), match.group(\'value\')\n                        if name:\n                            index = self.INT_ARGS_NAMES.index(name)\n                        arg_values[index] = value\n\n                    num_digits, minumum, maximum = arg_values\n                    schema = {\n                        \'type\': \'integer\',\n                        \'format\': f\'int{num_digits}\' if num_digits else \'int32\',\n                    }\n                    if minumum:\n                        schema[\'minimum\'] = minumum\n                    if maximum:\n                        schema[\'maximum\'] = maximum\n                elif converter == \'uuid\':\n                    schema = {\n                        \'type\': \'string\',\n                        \'format\': \'uuid\'\n                    }\n                elif converter == \'dt\':\n                    schema = {\n                        \'type\': \'string\',\n                        \'format\': \'date-time\',\n                    }\n                else:\n                    # no converter specified or customized converters\n                    schema = {\'type\': \'string\'}\n\n                parameters.append({\n                    \'name\': variable,\n                    \'in\': \'path\',\n                    \'required\': True,\n                    \'schema\': schema,\n                })\n\n        return f\'/{"/".join(subs)}\', parameters\n\n    def request_validation(self, req, query, json, headers, cookies):\n        if query:\n            req.context.query = query.parse_obj(req.params)\n        if headers:\n            req.context.headers = headers.parse_obj(req.headers)\n        if cookies:\n            req.context.cookies = cookies.parse_obj(req.cookies)\n        media = req.media or {}\n        if json:\n            req.context.json = json.parse_obj(media)\n\n    def validate(self,\n                 func,\n                 query, json, headers, cookies, resp,\n                 before, after,\n                 *args, **kwargs):\n        # falcon endpoint method arguments: (self, req, resp)\n        _self, _req, _resp = args[:3]\n        req_validation_error, resp_validation_error = None, None\n        try:\n            self.request_validation(_req, query, json, headers, cookies)\n\n        except ValidationError as err:\n            req_validation_error = err\n            _resp.status = \'422 Unprocessable Entity\'\n            _resp.media = err.errors()\n\n        before(_req, _resp, req_validation_error, _self)\n        if req_validation_error:\n            return\n\n        func(*args, **kwargs)\n        if resp and resp.has_model():\n            model = resp.find_model(_resp.status[:3])\n            if model:\n                try:\n                    model.validate(_resp.media)\n                except ValidationError as err:\n                    resp_validation_error = err\n                    _resp.status = \'500 Internal Service Response Validation Error\'\n                    _resp.media = err.errors()\n\n        after(_req, _resp, resp_validation_error, _self)\n\n    def bypass(self, func, method):\n        if not isinstance(func, partial):\n            return False\n        if inspect.ismethod(func.func):\n            return False\n        # others are <cyfunction>\n        return True\n\n```\n\n**File Path:** spectree/plugins/flask_plugin.py\n```python\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin, Context\nfrom .page import PAGES\n\n\nclass FlaskPlugin(BasePlugin):\n    blueprint_state = None\n\n    def find_routes(self):\n        from flask import current_app\n        if self.blueprint_state:\n            excludes = [f\'{self.blueprint_state.blueprint.name}.{ep}\'\n                        for ep in [\'static\', \'openapi\'] + [f\'doc_page_{ui}\' for ui in PAGES]]\n            for rule in current_app.url_map.iter_rules():\n                if self.blueprint_state.url_prefix and \\\n                        not str(rule).startswith(self.blueprint_state.url_prefix):\n                    continue\n                if rule.endpoint in excludes:\n                    continue\n                yield rule\n        else:\n            for rule in self.app.url_map.iter_rules():\n                if any(str(rule).startswith(path) for path in (\n                        f\'/{self.config.PATH}\', \'/static\'\n                )):\n                    continue\n                yield rule\n\n    def bypass(self, func, method):\n        if method in [\'HEAD\', \'OPTIONS\']:\n            return True\n        return False\n\n    def parse_func(self, route):\n        if self.blueprint_state:\n            func = self.blueprint_state.app.view_functions[route.endpoint]\n        else:\n            func = self.app.view_functions[route.endpoint]\n\n        for method in route.methods:\n            yield method, func\n\n    def parse_path(self, route):\n        from werkzeug.routing import parse_rule, parse_converter_args\n\n        subs = []\n        parameters = []\n\n        for converter, arguments, variable in parse_rule(str(route)):\n            if converter is None:\n                subs.append(variable)\n                continue\n            subs.append(f\'{{{variable}}}\')\n\n            args, kwargs = [], {}\n\n            if arguments:\n                args, kwargs = parse_converter_args(arguments)\n\n            schema = None\n            if converter == \'any\':\n                schema = {\n                    \'type\': \'array\',\n                    \'items\': {\n                        \'type\': \'string\',\n                        \'enum\': args,\n                    }\n                }\n            elif converter == \'int\':\n                schema = {\n                    \'type\': \'integer\',\n                    \'format\': \'int32\',\n                }\n                if \'max\' in kwargs:\n                    schema[\'maximum\'] = kwargs[\'max\']\n                if \'min\' in kwargs:\n                    schema[\'minimum\'] = kwargs[\'min\']\n            elif converter == \'float\':\n                schema = {\n                    \'type\': \'number\',\n                    \'format\': \'float\',\n                }\n            elif converter == \'uuid\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'uuid\',\n                }\n            elif converter == \'path\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'path\',\n                }\n            elif converter == \'string\':\n                schema = {\n                    \'type\': \'string\',\n                }\n                for prop in [\'length\', \'maxLength\', \'minLength\']:\n                    if prop in kwargs:\n                        schema[prop] = kwargs[prop]\n            elif converter == \'default\':\n                schema = {\'type\': \'string\'}\n\n            parameters.append({\n                \'name\': variable,\n                \'in\': \'path\',\n                \'required\': True,\n                \'schema\': schema,\n            })\n\n        return \'\'.join(subs), parameters\n\n    def request_validation(self, request, query, json, headers, cookies):\n        req_query = request.args or {}\n        req_json = request.get_json() or {}\n        req_headers = request.headers or {}\n        req_cookies = request.cookies or {}\n        request.context = Context(\n            query.parse_obj(req_query) if query else None,\n            json.parse_obj(req_json) if json else None,\n            headers.parse_obj(req_headers) if headers else None,\n            cookies.parse_obj(req_cookies) if cookies else None,\n        )\n\n    def validate(self,\n                 func,\n                 query, json, headers, cookies, resp,\n                 before, after,\n                 *args, **kwargs):\n        from flask import request, abort, make_response, jsonify\n\n        response, req_validation_error, resp_validation_error = None, None, None\n        try:\n            self.request_validation(request, query, json, headers, cookies)\n        except ValidationError as err:\n            req_validation_error = err\n            response = make_response(jsonify(err.errors()), 422)\n\n        before(request, response, req_validation_error, None)\n        if req_validation_error:\n            abort(response)\n\n        response = make_response(func(*args, **kwargs))\n\n        if resp and resp.has_model():\n            model = resp.find_model(response.status_code)\n            if model:\n                try:\n                    model.validate(response.get_json())\n                except ValidationError as err:\n                    resp_validation_error = err\n                    response = make_response(jsonify(\n                        {\'message\': \'response validation error\'}\n                    ), 500)\n\n        after(request, response, resp_validation_error, None)\n\n        return response\n\n    def register_route(self, app):\n        self.app = app\n        from flask import jsonify, Blueprint\n\n        self.app.add_url_rule(\n            self.config.spec_url,\n            \'openapi\',\n            lambda: jsonify(self.spectree.spec),\n        )\n\n        if isinstance(app, Blueprint):\n            def gen_doc_page(ui):\n                spec_url = self.config.spec_url\n                if self.blueprint_state.url_prefix is not None:\n                    spec_url = \'/\'.join((\n                        self.blueprint_state.url_prefix.rstrip(\'/\'),\n                        self.config.spec_url.lstrip(\'/\'))\n                    )\n\n                return PAGES[ui].format(spec_url)\n\n            for ui in PAGES:\n                app.add_url_rule(\n                    f\'/{self.config.PATH}/{ui}\',\n                    f\'doc_page_{ui}\',\n                    lambda ui=ui: gen_doc_page(ui)\n                )\n\n            app.record(lambda state: setattr(self, \'blueprint_state\', state))\n        else:\n            for ui in PAGES:\n                self.app.add_url_rule(\n                    f\'/{self.config.PATH}/{ui}\',\n                    f\'doc_page_{ui}\',\n                    lambda ui=ui: PAGES[ui].format(self.config.spec_url)\n                )\n\n```\n\n**File Path:** spectree/plugins/starlette_plugin.py\n```python\nimport inspect\nfrom collections import namedtuple\nfrom functools import partial\nfrom json import JSONDecodeError\nfrom json import loads as json_loads\n\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin, Context\nfrom .page import PAGES\n\nMETHODS = {\'get\', \'post\', \'put\', \'patch\', \'delete\'}\nRoute = namedtuple(\'Route\', [\'path\', \'methods\', \'func\'])\n\n\nclass StarlettePlugin(BasePlugin):\n    def __init__(self, spectree):\n        super().__init__(spectree)\n        from starlette.convertors import CONVERTOR_TYPES\n        self.conv2type = {\n            conv: typ for typ, conv in CONVERTOR_TYPES.items()\n        }\n\n    def register_route(self, app):\n        self.app = app\n        from starlette.responses import JSONResponse, HTMLResponse\n\n        self.app.add_route(\n            self.config.spec_url,\n            lambda request: JSONResponse(self.spectree.spec),\n        )\n\n        for ui in PAGES:\n            self.app.add_route(\n                f\'/{self.config.PATH}/{ui}\',\n                lambda request, ui=ui: HTMLResponse(\n                    PAGES[ui].format(self.config.spec_url)\n                ),\n            )\n\n    async def request_validation(self, request, query, json, headers, cookies):\n        request.context = Context(\n            query.parse_obj(request.query_params) if query else None,\n            json.parse_obj(json_loads(await request.body() or \'{}\')) if json else None,\n            headers.parse_obj(request.headers) if headers else None,\n            cookies.parse_obj(request.cookies) if cookies else None,\n        )\n\n    async def validate(self,\n                       func,\n                       query, json, headers, cookies, resp,\n                       before, after,\n                       *args, **kwargs):\n        from starlette.responses import JSONResponse\n\n        # NOTE: If func is a `HTTPEndpoint`, it should have \'.\' in its ``__qualname__``\n        # This is not elegant. But it seems `inspect` doesn\'t work here.\n        instance = args[0] if \'.\' in func.__qualname__ else None\n        request = args[1] if \'.\' in func.__qualname__ else args[0]\n        response = None\n        req_validation_error, resp_validation_error, json_decode_error = None, None, None\n\n        try:\n            await self.request_validation(request, query, json, headers, cookies)\n        except ValidationError as err:\n            req_validation_error = err\n            response = JSONResponse(err.errors(), 422)\n        except JSONDecodeError as err:\n            json_decode_error = err\n            self.logger.info(\n                \'422 Validation Error\',\n                extra={\'spectree_json_decode_error\': str(err)}\n            )\n            response = JSONResponse({\'error_msg\': str(err)}, 422)\n\n        before(request, response, req_validation_error, instance)\n        if req_validation_error or json_decode_error:\n            return response\n\n        if inspect.iscoroutinefunction(func):\n            response = await func(*args, **kwargs)\n        else:\n            response = func(*args, **kwargs)\n\n        if resp:\n            model = resp.find_model(response.status_code)\n            if model:\n                try:\n                    model.validate(json_loads(response.body))\n                except ValidationError as err:\n                    resp_validation_error = err\n                    response = JSONResponse(err.errors(), 500)\n\n        after(request, response, resp_validation_error, instance)\n\n        return response\n\n    def find_routes(self):\n        routes = []\n\n        def parse_route(app, prefix=\'\'):\n            for route in app.routes:\n                if route.path.startswith(f\'/{self.config.PATH}\'):\n                    continue\n\n                func = route.app\n                if isinstance(func, partial):\n                    try:\n                        func = func.__wrapped__\n                    except AttributeError:\n                        pass\n\n                if inspect.isclass(func):\n                    for method in METHODS:\n                        if getattr(func, method, None):\n                            routes.append(Route(\n                                f\'{prefix}{route.path}\',\n                                {method.upper()},\n                                getattr(func, method)\n                            ))\n                elif inspect.isfunction(func):\n                    routes.append(Route(\n                        f\'{prefix}{route.path}\',\n                        route.methods,\n                        route.endpoint))\n                else:\n                    parse_route(route, prefix=f\'{prefix}{route.path}\')\n\n        parse_route(self.app)\n        return routes\n\n    def bypass(self, func, method):\n        if method in [\'HEAD\', \'OPTIONS\']:\n            return True\n        return False\n\n    def parse_func(self, route):\n        for method in route.methods or [\'GET\']:\n            yield method, route.func\n\n    def parse_path(self, route):\n        from starlette.routing import compile_path\n        _, path, variables = compile_path(route.path)\n        parameters = []\n\n        for name, conv in variables.items():\n            schema = None\n            typ = self.conv2type[conv]\n            if typ == \'int\':\n                schema = {\n                    \'type\': \'integer\',\n                    \'format\': \'int32\'\n                }\n            elif typ == \'float\':\n                schema = {\n                    \'type\': \'number\',\n                    \'format\': \'float\',\n                }\n            elif typ == \'path\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'path\',\n                }\n            elif typ == \'str\':\n                schema = {\'type\': \'string\'}\n\n            parameters.append({\n                \'name\': name,\n                \'in\': \'path\',\n                \'required\': True,\n                \'schema\': schema,\n            })\n\n        return path, parameters\n\n```\n\n**File Path:** spectree/plugins/falcon_plugin.py\n```python\nimport inspect\nimport re\nfrom functools import partial\n\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin\nfrom .page import PAGES\n\n\nclass OpenAPI:\n    def __init__(self, spec):\n        self.spec = spec\n\n    def on_get(self, req, resp):\n        resp.media = self.spec\n\n\nclass DocPage:\n    def __init__(self, html, spec_url):\n        self.page = html.format(spec_url)\n\n    def on_get(self, req, resp):\n        resp.content_type = \'text/html\'\n        resp.body = self.page\n\n\nDOC_CLASS = [x.__name__ for x in (DocPage, OpenAPI)]\n\n\nclass FalconPlugin(BasePlugin):\n    def __init__(self, spectree):\n        super().__init__(spectree)\n        from falcon.routing.compiled import _FIELD_PATTERN\n\n        self.FIELD_PATTERN = _FIELD_PATTERN\n        # NOTE from `falcon.routing.compiled.CompiledRouterNode`\n        self.ESCAPE = r\'[\\.\\(\\)\\[\\]\\?\\$\\*\\+\\^\\|]\'\n        self.ESCAPE_TO = r\'\\\\\\g<0>\'\n        self.EXTRACT = r\'{\\2}\'\n        # NOTE this regex is copied from werkzeug.routing._converter_args_re and\n        # modified to support only int args\n        self.INT_ARGS = re.compile(r"""\n            ((?P<name>\\w+)\\s*=\\s*)?\n            (?P<value>\\d+)\\s*\n        """, re.VERBOSE)\n        self.INT_ARGS_NAMES = (\'num_digits\', \'min\', \'max\')\n\n    def register_route(self, app):\n        self.app = app\n        self.app.add_route(\n            self.config.spec_url, OpenAPI(self.spectree.spec)\n        )\n        for ui in PAGES:\n            self.app.add_route(\n                f\'/{self.config.PATH}/{ui}\',\n                DocPage(PAGES[ui], self.config.spec_url),\n            )\n\n    def find_routes(self):\n        routes = []\n\n        def find_node(node):\n            if node.resource and node.resource.__class__.__name__ not in DOC_CLASS:\n                routes.append(node)\n\n            for child in node.children:\n                find_node(child)\n\n        for route in self.app._router._roots:\n            find_node(route)\n\n        return routes\n\n    def parse_func(self, route):\n        return route.method_map.items()\n\n    def parse_path(self, route):\n        subs, parameters = [], []\n        for segment in route.uri_template.strip(\'/\').split(\'/\'):\n            matches = self.FIELD_PATTERN.finditer(segment)\n            if not matches:\n                subs.append(segment)\n                continue\n\n            escaped = re.sub(self.ESCAPE, self.ESCAPE_TO, segment)\n            subs.append(self.FIELD_PATTERN.sub(self.EXTRACT, escaped))\n\n            for field in matches:\n                variable, converter, argstr = [field.group(name) for name in\n                                               (\'fname\', \'cname\', \'argstr\')]\n\n                if converter == \'int\':\n                    if argstr is None:\n                        argstr = \'\'\n\n                    arg_values = [None, None, None]\n                    for index, match in enumerate(self.INT_ARGS.finditer(argstr)):\n                        name, value = match.group(\'name\'), match.group(\'value\')\n                        if name:\n                            index = self.INT_ARGS_NAMES.index(name)\n                        arg_values[index] = value\n\n                    num_digits, minumum, maximum = arg_values\n                    schema = {\n                        \'type\': \'integer\',\n                        \'format\': f\'int{num_digits}\' if num_digits else \'int32\',\n                    }\n                    if minumum:\n                        schema[\'minimum\'] = minumum\n                    if maximum:\n                        schema[\'maximum\'] = maximum\n                elif converter == \'uuid\':\n                    schema = {\n                        \'type\': \'string\',\n                        \'format\': \'uuid\'\n                    }\n                elif converter == \'dt\':\n                    schema = {\n                        \'type\': \'string\',\n                        \'format\': \'date-time\',\n                    }\n                else:\n                    # no converter specified or customized converters\n                    schema = {\'type\': \'string\'}\n\n                parameters.append({\n                    \'name\': variable,\n                    \'in\': \'path\',\n                    \'required\': True,\n                    \'schema\': schema,\n                })\n\n        return f\'/{"/".join(subs)}\', parameters\n\n    def request_validation(self, req, query, json, headers, cookies):\n        if query:\n            req.context.query = query.parse_obj(req.params)\n        if headers:\n            req.context.headers = headers.parse_obj(req.headers)\n        if cookies:\n            req.context.cookies = cookies.parse_obj(req.cookies)\n        media = req.media or {}\n        if json:\n            req.context.json = json.parse_obj(media)\n\n    def validate(self,\n                 func,\n                 query, json, headers, cookies, resp,\n                 before, after,\n                 *args, **kwargs):\n        # falcon endpoint method arguments: (self, req, resp)\n        _self, _req, _resp = args[:3]\n        req_validation_error, resp_validation_error = None, None\n        try:\n            self.request_validation(_req, query, json, headers, cookies)\n\n        except ValidationError as err:\n            req_validation_error = err\n            _resp.status = \'422 Unprocessable Entity\'\n            _resp.media = err.errors()\n\n        before(_req, _resp, req_validation_error, _self)\n        if req_validation_error:\n            return\n\n        func(*args, **kwargs)\n        if resp and resp.has_model():\n            model = resp.find_model(_resp.status[:3])\n            if model:\n                try:\n                    model.validate(_resp.media)\n                except ValidationError as err:\n                    resp_validation_error = err\n                    _resp.status = \'500 Internal Service Response Validation Error\'\n                    _resp.media = err.errors()\n\n        after(_req, _resp, resp_validation_error, _self)\n\n    def bypass(self, func, method):\n        if not isinstance(func, partial):\n            return False\n        if inspect.ismethod(func.func):\n            return False\n        # others are <cyfunction>\n        return True\n\n```\n\n**File Path:** spectree/plugins/flask_plugin.py\n```python\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin, Context\nfrom .page import PAGES\n\n\nclass FlaskPlugin(BasePlugin):\n    blueprint_state = None\n\n    def find_routes(self):\n        from flask import current_app\n        if self.blueprint_state:\n            excludes = [f\'{self.blueprint_state.blueprint.name}.{ep}\'\n                        for ep in [\'static\', \'openapi\'] + [f\'doc_page_{ui}\' for ui in PAGES]]\n            for rule in current_app.url_map.iter_rules():\n                if self.blueprint_state.url_prefix and \\\n                        not str(rule).startswith(self.blueprint_state.url_prefix):\n                    continue\n                if rule.endpoint in excludes:\n                    continue\n                yield rule\n        else:\n            for rule in self.app.url_map.iter_rules():\n                if any(str(rule).startswith(path) for path in (\n                        f\'/{self.config.PATH}\', \'/static\'\n                )):\n                    continue\n                yield rule\n\n    def bypass(self, func, method):\n        if method in [\'HEAD\', \'OPTIONS\']:\n            return True\n        return False\n\n    def parse_func(self, route):\n        if self.blueprint_state:\n            func = self.blueprint_state.app.view_functions[route.endpoint]\n        else:\n            func = self.app.view_functions[route.endpoint]\n\n        for method in route.methods:\n            yield method, func\n\n    def parse_path(self, route):\n        from werkzeug.routing import parse_rule, parse_converter_args\n\n        subs = []\n        parameters = []\n\n        for converter, arguments, variable in parse_rule(str(route)):\n            if converter is None:\n                subs.append(variable)\n                continue\n            subs.append(f\'{{{variable}}}\')\n\n            args, kwargs = [], {}\n\n            if arguments:\n                args, kwargs = parse_converter_args(arguments)\n\n            schema = None\n            if converter == \'any\':\n                schema = {\n                    \'type\': \'array\',\n                    \'items\': {\n                        \'type\': \'string\',\n                        \'enum\': args,\n                    }\n                }\n            elif converter == \'int\':\n                schema = {\n                    \'type\': \'integer\',\n                    \'format\': \'int32\',\n                }\n                if \'max\' in kwargs:\n                    schema[\'maximum\'] = kwargs[\'max\']\n                if \'min\' in kwargs:\n                    schema[\'minimum\'] = kwargs[\'min\']\n            elif converter == \'float\':\n                schema = {\n                    \'type\': \'number\',\n                    \'format\': \'float\',\n                }\n            elif converter == \'uuid\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'uuid\',\n                }\n            elif converter == \'path\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'path\',\n                }\n            elif converter == \'string\':\n                schema = {\n                    \'type\': \'string\',\n                }\n                for prop in [\'length\', \'maxLength\', \'minLength\']:\n                    if prop in kwargs:\n                        schema[prop] = kwargs[prop]\n            elif converter == \'default\':\n                schema = {\'type\': \'string\'}\n\n            parameters.append({\n                \'name\': variable,\n                \'in\': \'path\',\n                \'required\': True,\n                \'schema\': schema,\n            })\n\n        return \'\'.join(subs), parameters\n\n    def request_validation(self, request, query, json, headers, cookies):\n        req_query = request.args or {}\n        req_json = request.get_json() or {}\n        req_headers = request.headers or {}\n        req_cookies = request.cookies or {}\n        request.context = Context(\n            query.parse_obj(req_query) if query else None,\n            json.parse_obj(req_json) if json else None,\n            headers.parse_obj(req_headers) if headers else None,\n            cookies.parse_obj(req_cookies) if cookies else None,\n        )\n\n    def validate(self,\n                 func,\n                 query, json, headers, cookies, resp,\n                 before, after,\n                 *args, **kwargs):\n        from flask import request, abort, make_response, jsonify\n\n        response, req_validation_error, resp_validation_error = None, None, None\n        try:\n            self.request_validation(request, query, json, headers, cookies)\n        except ValidationError as err:\n            req_validation_error = err\n            response = make_response(jsonify(err.errors()), 422)\n\n        before(request, response, req_validation_error, None)\n        if req_validation_error:\n            abort(response)\n\n        response = make_response(func(*args, **kwargs))\n\n        if resp and resp.has_model():\n            model = resp.find_model(response.status_code)\n            if model:\n                try:\n                    model.validate(response.get_json())\n                except ValidationError as err:\n                    resp_validation_error = err\n                    response = make_response(jsonify(\n                        {\'message\': \'response validation error\'}\n                    ), 500)\n\n        after(request, response, resp_validation_error, None)\n\n        return response\n\n    def register_route(self, app):\n        self.app = app\n        from flask import jsonify, Blueprint\n\n        self.app.add_url_rule(\n            self.config.spec_url,\n            \'openapi\',\n            lambda: jsonify(self.spectree.spec),\n        )\n\n        if isinstance(app, Blueprint):\n            def gen_doc_page(ui):\n                spec_url = self.config.spec_url\n                if self.blueprint_state.url_prefix is not None:\n                    spec_url = \'/\'.join((\n                        self.blueprint_state.url_prefix.rstrip(\'/\'),\n                        self.config.spec_url.lstrip(\'/\'))\n                    )\n\n                return PAGES[ui].format(spec_url)\n\n            for ui in PAGES:\n                app.add_url_rule(\n                    f\'/{self.config.PATH}/{ui}\',\n                    f\'doc_page_{ui}\',\n                    lambda ui=ui: gen_doc_page(ui)\n                )\n\n            app.record(lambda state: setattr(self, \'blueprint_state\', state))\n        else:\n            for ui in PAGES:\n                self.app.add_url_rule(\n                    f\'/{self.config.PATH}/{ui}\',\n                    f\'doc_page_{ui}\',\n                    lambda ui=ui: PAGES[ui].format(self.config.spec_url)\n                )\n\n```\n\n\n\n## Bug Example 2:\n**Problem Statement:** Environment variable GITHUB_TOKEN is ignored\nIf `--token` isn\'t provided on invocation while environment variable `GITHUB_TOKEN` is defined, the environment variable\'s value should be used as the access token.\n\n**Modified Files:**\n**File Path:** pip_install_privates/install.py\n```python\n#!/usr/bin/env python\nimport argparse\nimport os\nfrom pip import __version__ as pip_version\n\nfrom pip_install_privates.utils import parse_pip_version\n\npip_version_tuple = parse_pip_version(pip_version)\ngte_18_1 = pip_version_tuple[0] == 18 and pip_version_tuple[1] >= 1\nif pip_version_tuple[0] > 18 or gte_18_1:\n    from pip._internal import main as pip_main\n    from pip._internal.cli import status_codes\n\nelif pip_version_tuple[0] >= 10:\n    from pip._internal import status_codes, main as pip_main\n\nelse:\n    from pip import status_codes, main as pip_main\n\n\ndef convert_url(url, token):\n    if url.startswith(\'git+ssh://git@github.com/\'):\n        return \'git+https://{}:x-oauth-basic@github.com/{}\'.format(token, url[25:])\n    elif url.startswith(\'git+git@github.com:\'):\n        return \'git+https://{}:x-oauth-basic@github.com/{}\'.format(token, url[19:])\n    return url\n\ndef collect_requirements(fname, transform_with_token=None):\n    with open(fname) as reqs:\n        contents = reqs.readlines()\n\n    collected = []\n    for line in contents:\n        line = line.strip()\n\n        if not line or line.startswith(\'#\'):\n            continue\n\n        tokens = line.split()\n\n        # Handles:\n        #   alembic>=0.8\n        #   alembic==0.8.8\n        #   alembic==0.8.8  # so we can apply Hypernode/ByteDB fixtures\n        #   git+git://github.com/myself/myproject\n        #   git+ssh://github.com/myself/myproject@v2\n        #\n        if len(tokens) == 1 or tokens[1].startswith(\'#\'):\n            if (tokens[0].startswith(\'git+ssh\') or tokens[0].startswith(\'git+git\')) and transform_with_token:\n                collected.append(convert_url(tokens[0], transform_with_token))\n            else:\n                collected.append(tokens[0])\n\n\n        # Handles:\n        #   -r base.txt\n        elif tokens[0] == \'-r\':\n            curdir = os.path.abspath(os.path.dirname(fname))\n            collected += collect_requirements(os.path.join(curdir, tokens[1]),\n                                              transform_with_token=transform_with_token)\n\n        # Rewrite private repositories that normally would use ssh (with keys in an agent), to using\n        # an oauth key\n        elif tokens[0] == \'-e\':\n            if tokens[1].startswith(\'git+ssh\') or tokens[1].startswith(\'git+git\'):\n                if transform_with_token:\n                    collected.append(convert_url(tokens[1], transform_with_token))\n                else:\n                    collected.append(\'-e {}\'.format(tokens[1]))\n            else:\n                # Strip development flag `-e` to prevent dependencies installed within the project\n                collected += [tokens[1]]\n\n        # No special casing for the rest. Just pass everything to pip\n        else:\n            collected += tokens\n\n    return collected\n\n\ndef install():\n    parser = argparse.ArgumentParser(\n        formatter_class=argparse.RawDescriptionHelpFormatter,\n        description="""\nInstall all requirements from specified file with pip. Optionally transform\ngit+git and git+ssh url to private repo\'s to use a given Personal Access Token for\ngithub. That way installing them does not depend on a ssh-agent with suitable\nkeys. Which you don\'t have when installing requirements in a Docker.\nThese URLs will also be stripped of the -e flag, so they\'re installed globally.\nNote the -e flag is optional for the git+git//github.com and git+ssh://github.com\nurls.\n\nThis means that the following URL:\n  -e git+git@github.com:MyOrg/my-project.git@my-tag#egg=my_project\nwould be transformed to:\n  git+https://<token>:x-oauth-basic@github.com/MyOrg/my-project.git@my-tag#egg=my_project\n\nNon-private GitHub URL\'s (git+https) and non-GitHub URL\'s are kept as-is, but\nare also stripped of the -e flag. If no token is given, private URLs will be\nkept, including the -e flag (otherwise they can\'t be installed at all).\n""")\n\n    parser.add_argument(\'--token\', \'-t\', help=\'Your Personal Access Token for private GitHub repositories\',\n                        default=os.environ.get(\'GITHUB_TOKEN\'))\n    parser.add_argument(\'req_file\', help=\'path to the requirements file to install\')\n    args = parser.parse_args()\n\n    # TODO: rewrite to a clear collect and a clear transform phase. Or pass in a transform function\n    pip_args = [\'install\'] + collect_requirements(args.req_file, transform_with_token=args.token)\n    if pip_main(pip_args) != status_codes.SUCCESS:\n        raise RuntimeError(\'Error installing requirements\')\n\n\nif __name__ == \'__main__\':\n    install()\n\n```\n\n**File Path:** setup.py\n```python\n#!/usr/bin/env python\nfrom setuptools import setup\nfrom os.path import abspath, dirname, join\n\n\ndef readfile(filename):\n    path = join(dirname(abspath(__file__)), filename)\n    with open(path, \'rt\') as filehandle:\n        return filehandle.read()\n\n\nsetup(\n    name=\'pip_install_privates\',\n    version=\'0.5.2\',\n    description=\'Install pip packages from private repositories without an ssh agent\',\n    long_description=readfile(\'README.rst\'),\n    long_description_content_type=\'text/x-rst\',\n    author=\'Byte Internet\',\n    author_email=\'tech@byte.nl\',\n    license=\'MIT\',\n    url=\'https://github.com/ByteInternet/pip-install-privates\',\n    packages=[\'pip_install_privates\'],\n    install_requires=[\'pip\'],\n    entry_points={\n        \'console_scripts\': [\n            \'pip_install_privates = pip_install_privates.install:install\'\n        ]\n    }\n)\n\n```\n\n\n\n## Bug Example 3:\n**Problem Statement:** slicing a Batch returns a list\nShould return a Batch instead.\r\n\r\n```\r\n>>> import hyp3_sdk\r\n>>> hyp3 = hyp3_sdk.HyP3()\r\n>>> jobs = hyp3.find_jobs()\r\n>>> type(jobs)\r\n<class \'hyp3_sdk.jobs.Batch\'>\r\n>>> len(jobs)\r\n955\r\n>>> type(jobs[3:10])\r\n<class \'list\'>\r\n```\n\n**Modified Files:**\n**File Path:** hyp3_sdk/jobs.py\n```python\nfrom collections import Counter\nfrom datetime import datetime\nfrom pathlib import Path\nfrom typing import List, Optional, Union\n\nfrom dateutil import tz\nfrom dateutil.parser import parse as parse_date\nfrom requests import HTTPError\n\nfrom hyp3_sdk.exceptions import HyP3SDKError\nfrom hyp3_sdk.util import download_file, get_tqdm_progress_bar\n\n\n# TODO: actually looks like a good candidate for a dataclass (python 3.7+)\n#       https://docs.python.org/3/library/dataclasses.html\nclass Job:\n    _attributes_for_resubmit = {\'name\', \'job_parameters\', \'job_type\'}\n\n    def __init__(\n            self,\n            job_type: str,\n            job_id: str,\n            request_time: datetime,\n            status_code: str,\n            user_id: str,\n            name: Optional[str] = None,\n            job_parameters: Optional[dict] = None,\n            files: Optional[List] = None,\n            logs: Optional[List] = None,\n            browse_images: Optional[List] = None,\n            thumbnail_images: Optional[List] = None,\n            expiration_time: Optional[datetime] = None\n    ):\n        self.job_id = job_id\n        self.job_type = job_type\n        self.request_time = request_time\n        self.status_code = status_code\n        self.user_id = user_id\n        self.name = name\n        self.job_parameters = job_parameters\n        self.files = files\n        self.logs = logs\n        self.browse_images = browse_images\n        self.thumbnail_images = thumbnail_images\n        self.expiration_time = expiration_time\n\n    def __repr__(self):\n        return f\'Job.from_dict({self.to_dict()})\'\n\n    def __str__(self):\n        return f\'HyP3 {self.job_type} job {self.job_id}\'\n\n    def __eq__(self, other):\n        return self.__dict__ == other.__dict__\n\n    @staticmethod\n    def from_dict(input_dict: dict):\n        expiration_time = parse_date(input_dict[\'expiration_time\']) if input_dict.get(\'expiration_time\') else None\n        return Job(\n            job_type=input_dict[\'job_type\'],\n            job_id=input_dict[\'job_id\'],\n            request_time=parse_date(input_dict[\'request_time\']),\n            status_code=input_dict[\'status_code\'],\n            user_id=input_dict[\'user_id\'],\n            name=input_dict.get(\'name\'),\n            job_parameters=input_dict.get(\'job_parameters\'),\n            files=input_dict.get(\'files\'),\n            logs=input_dict.get(\'logs\'),\n            browse_images=input_dict.get(\'browse_images\'),\n            thumbnail_images=input_dict.get(\'thumbnail_images\'),\n            expiration_time=expiration_time\n        )\n\n    def to_dict(self, for_resubmit: bool = False):\n        job_dict = {}\n        if for_resubmit:\n            keys_to_process = Job._attributes_for_resubmit\n        else:\n            keys_to_process = vars(self).keys()\n\n        for key in keys_to_process:\n            value = self.__getattribute__(key)\n            if value is not None:\n                if isinstance(value, datetime):\n                    job_dict[key] = value.isoformat(timespec=\'seconds\')\n                else:\n                    job_dict[key] = value\n\n        return job_dict\n\n    def succeeded(self) -> bool:\n        return self.status_code == \'SUCCEEDED\'\n\n    def failed(self) -> bool:\n        return self.status_code == \'FAILED\'\n\n    def complete(self) -> bool:\n        return self.succeeded() or self.failed()\n\n    def running(self) -> bool:\n        return not self.complete()\n\n    def expired(self) -> bool:\n        try:\n            return datetime.now(tz.UTC) >= self.expiration_time\n        except TypeError:\n            raise HyP3SDKError(\'Only SUCCEEDED jobs have an expiration time\')\n\n    def download_files(self, location: Union[Path, str] = \'.\', create: bool = True) -> List[Path]:\n        """\n        Args:\n            location: Directory location to put files into\n            create: Create `location` if it does not point to an existing directory\n\n        Returns: list of Path objects to downloaded files\n        """\n        location = Path(location)\n\n        if not self.succeeded():\n            raise HyP3SDKError(f\'Only succeeded jobs can be downloaded; job is {self.status_code}.\')\n        if self.expired():\n            raise HyP3SDKError(f\'Expired jobs cannot be downloaded; \'\n                               f\'job expired {self.expiration_time.isoformat(timespec="seconds")}.\')\n\n        if create:\n            location.mkdir(parents=True, exist_ok=True)\n        elif not location.is_dir():\n            raise NotADirectoryError(str(location))\n\n        downloaded_files = []\n        for file in self.files:\n            download_url = file[\'url\']\n            filename = location / file[\'filename\']\n            try:\n                downloaded_files.append(download_file(download_url, filename, chunk_size=10485760))\n            except HTTPError:\n                raise HyP3SDKError(f\'Unable to download file: {download_url}\')\n        return downloaded_files\n\n\nclass Batch:\n    def __init__(self, jobs: Optional[List[Job]] = None):\n        if jobs is None:\n            jobs = []\n        self.jobs = jobs\n\n    def __add__(self, other: Union[Job, \'Batch\']):\n        if isinstance(other, Batch):\n            return Batch(self.jobs + other.jobs)\n        elif isinstance(other, Job):\n            return Batch(self.jobs + [other])\n        else:\n            raise TypeError(f"unsupported operand type(s) for +: \'{type(self)}\' and \'{type(other)}\'")\n\n    def __iadd__(self, other: Union[Job, \'Batch\']):\n        if isinstance(other, Batch):\n            self.jobs += other.jobs\n        elif isinstance(other, Job):\n            self.jobs += [other]\n        else:\n            raise TypeError(f"unsupported operand type(s) for +=: \'{type(self)}\' and \'{type(other)}\'")\n        return self\n\n    def __iter__(self):\n        return iter(self.jobs)\n\n    def __len__(self):\n        return len(self.jobs)\n\n    def __contains__(self, job: Job):\n        return job in self.jobs\n\n    def __eq__(self, other: \'Batch\'):\n        return self.jobs == other.jobs\n\n    def __delitem__(self, job: int):\n        self.jobs.pop(job)\n        return self\n\n    def __getitem__(self, index: int):\n        if isinstance(index, slice):\n            return Batch(self.jobs[index])\n        return self.jobs[index]\n\n    def __setitem__(self, index: int, job: Job):\n        self.jobs[index] = job\n        return self\n\n    def __repr__(self):\n        reprs = ", ".join([job.__repr__() for job in self.jobs])\n        return f\'Batch([{reprs}])\'\n\n    def __str__(self):\n        count = self._count_statuses()\n        return f\'{len(self)} HyP3 Jobs: \' \\\n               f\'{count["SUCCEEDED"]} succeeded, \' \\\n               f\'{count["FAILED"]} failed, \' \\\n               f\'{count["RUNNING"]} running, \' \\\n               f\'{count["PENDING"]} pending.\'\n\n    def _count_statuses(self):\n        return Counter([job.status_code for job in self.jobs])\n\n    def complete(self) -> bool:\n        """\n        Returns: True if all jobs are complete, otherwise returns False\n        """\n        for job in self.jobs:\n            if not job.complete():\n                return False\n        return True\n\n    def succeeded(self) -> bool:\n        """\n        Returns: True if all jobs have succeeded, otherwise returns False\n        """\n        for job in self.jobs:\n            if not job.succeeded():\n                return False\n        return True\n\n    def download_files(self, location: Union[Path, str] = \'.\', create: bool = True) -> List[Path]:\n        """\n        Args:\n            location: Directory location to put files into\n            create: Create `location` if it does not point to an existing directory\n\n        Returns: list of Path objects to downloaded files\n        """\n        downloaded_files = []\n        tqdm = get_tqdm_progress_bar()\n        for job in tqdm(self.jobs):\n            try:\n                downloaded_files.extend(job.download_files(location, create))\n            except HyP3SDKError as e:\n                print(f\'Warning: {e}. Skipping download for {job}.\')\n        return downloaded_files\n\n    def any_expired(self) -> bool:\n        """Check succeeded jobs for expiration"""\n        for job in self.jobs:\n            try:\n                if job.expired():\n                    return True\n            except HyP3SDKError:\n                continue\n        return False\n\n    def filter_jobs(\n            self, succeeded: bool = True, running: bool = True, failed: bool = False, include_expired: bool = True,\n    ) -> \'Batch\':\n        """Filter jobs by status. By default, only succeeded and still running jobs will be in the returned batch.\n\n        Args:\n            succeeded: Include all succeeded jobs\n            running: Include all running jobs\n            failed: Include all failed jobs\n            include_expired: Include expired jobs in the result\n\n\n        Returns:\n             batch: A batch object containing jobs matching all the selected statuses\n        """\n        filtered_jobs = []\n\n        for job in self.jobs:\n            if job.succeeded() and succeeded:\n                if include_expired or not job.expired():\n                    filtered_jobs.append(job)\n\n            elif job.running() and running:\n                filtered_jobs.append(job)\n\n            elif job.failed() and failed:\n                filtered_jobs.append(job)\n\n        return Batch(filtered_jobs)\n\n```\n\n**File Path:** hyp3_sdk/util.py\n```python\n"""Extra utilities for working with HyP3"""\nfrom pathlib import Path\nfrom typing import Any, Generator, Sequence, Union\nfrom zipfile import ZipFile\n\nimport requests\nfrom requests.adapters import HTTPAdapter\nfrom urllib3.util.retry import Retry\n\nimport hyp3_sdk\nfrom hyp3_sdk.exceptions import AuthenticationError\n\nAUTH_URL = \'https://urs.earthdata.nasa.gov/oauth/authorize?response_type=code&client_id=BO_n7nTIlMljdvU6kRRB3g\' \\\n           \'&redirect_uri=https://auth.asf.alaska.edu/login&app_type=401\'\n\n\ndef extract_zipped_product(zip_file: Union[str, Path], delete: bool = True) -> Path:\n    """Extract a zipped HyP3 product\n\n    Extract a zipped HyP3 product to the same directory as the zipped HyP3 product, optionally\n    deleting `zip file` afterward.\n\n    Args:\n        zip_file: Zipped HyP3 product to extract\n        delete: Delete `zip_file` after it has been extracted\n\n    Returns:\n        Path to the HyP3 product folder containing the product files\n    """\n    zip_file = Path(zip_file)\n    with ZipFile(zip_file) as z:\n        z.extractall(path=zip_file.parent)\n\n    if delete:\n        zip_file.unlink()\n\n    return zip_file.parent / zip_file.stem\n\n\ndef chunk(itr: Sequence[Any], n: int = 200) -> Generator[Sequence[Any], None, None]:\n    """Split a sequence into small chunks\n\n    Args:\n        itr: A sequence object to chunk\n        n: Size of the chunks to return\n    """\n    if not isinstance(n, int) or n < 1:\n        raise ValueError(f\'n must be a positive integer: {n}\')\n\n    for i in range(0, len(itr), n):\n        yield itr[i:i + n]\n\n\ndef get_tqdm_progress_bar():\n    try:\n        # https://github.com/ASFHyP3/hyp3-sdk/issues/92\n        import ipywidgets  # noqa: F401\n        from tqdm.auto import tqdm\n    except ImportError:\n        from tqdm.std import tqdm\n    return tqdm\n\n\ndef get_authenticated_session(username: str, password: str) -> requests.Session:\n    """Log into HyP3 using credentials for `urs.earthdata.nasa.gov` from either the provided\n     credentials or a `.netrc` file.\n\n    Returns:\n        An authenticated HyP3 Session\n    """\n    s = requests.Session()\n    if hyp3_sdk.TESTING:\n        return s\n    if username is not None and password is not None:\n        response = s.get(AUTH_URL, auth=(username, password))\n        try:\n            response.raise_for_status()\n        except requests.HTTPError:\n            raise AuthenticationError(\'Was not able to authenticate with credentials provided\\n\'\n                                      \'This could be due to invalid credentials or a connection error.\')\n    else:\n        response = s.get(AUTH_URL)\n        try:\n            response.raise_for_status()\n        except requests.HTTPError:\n            raise AuthenticationError(\'Was not able to authenticate with .netrc file and no credentials provided\\n\'\n                                      \'This could be due to invalid credentials in .netrc or a connection error.\')\n    return s\n\n\ndef download_file(url: str, filepath: Union[Path, str], chunk_size=None, retries=2, backoff_factor=1) -> Path:\n    """Download a file\n    Args:\n        url: URL of the file to download\n        filepath: Location to place file into\n        chunk_size: Size to chunk the download into\n        retries: Number of retries to attempt\n        backoff_factor: Factor for calculating time between retries\n    Returns:\n        download_path: The path to the downloaded file\n    """\n    filepath = Path(filepath)\n    session = requests.Session()\n    retry_strategy = Retry(\n        total=retries,\n        backoff_factor=backoff_factor,\n        status_forcelist=[429, 500, 502, 503, 504],\n    )\n\n    session.mount(\'https://\', HTTPAdapter(max_retries=retry_strategy))\n    session.mount(\'http://\', HTTPAdapter(max_retries=retry_strategy))\n    stream = False if chunk_size is None else True\n    with session.get(url, stream=stream) as s:\n        s.raise_for_status()\n        tqdm = get_tqdm_progress_bar()\n        with tqdm.wrapattr(open(filepath, "wb"), \'write\', miniters=1, desc=filepath.name,\n                           total=int(s.headers.get(\'content-length\', 0))) as f:\n            for chunk in s.iter_content(chunk_size=chunk_size):\n                if chunk:\n                    f.write(chunk)\n    session.close()\n\n    return filepath\n\n```\n\n\n\n## Target Code (Original Clean Code):\nFile Name: spectree/utils.py\n\nFile Content:\n ```python\nimport re\nimport inspect\nimport logging\n\n# parse HTTP status code to get the code\nHTTP_CODE = re.compile(r\'^HTTP_(?P<code>\\d{3})$\')\n\nlogger = logging.getLogger(__name__)\n\n\ndef parse_comments(func):\n    """\n    parse function comments\n\n    First line of comments will be saved as summary, and the rest\n    will be saved as description.\n    """\n    doc = inspect.getdoc(func)\n    if doc is None:\n        return None, None\n    doc = doc.split(\'\\n\', 1)\n    if len(doc) == 1:\n        return doc[0], None\n    return doc[0], doc[1].strip()\n\n\ndef parse_request(func):\n    """\n    get json spec\n    """\n    data = {}\n    if hasattr(func, \'json\'):\n        data = {\n            \'content\': {\n                \'application/json\': {\n                    \'schema\': {\n                        \'$ref\': f\'#/components/schemas/{func.json}\'\n                    }\n                }\n            }\n        }\n    return data\n\n\ndef parse_params(func, params, models):\n    """\n    get spec for (query, headers, cookies)\n    """\n    if hasattr(func, \'query\'):\n        query = models[func.query]\n        for name, schema in query[\'properties\'].items():\n            params.append({\n                \'name\': name,\n                \'in\': \'query\',\n                \'schema\': schema,\n                \'required\': name in query.get(\'required\', []),\n                \'description\': schema.get(\'description\', \'\'),\n            })\n\n    if hasattr(func, \'headers\'):\n        headers = models[func.headers]\n        for name, schema in headers[\'properties\'].items():\n            params.append({\n                \'name\': name,\n                \'in\': \'header\',\n                \'schema\': schema,\n                \'required\': name in headers.get(\'required\', []),\n                \'description\': schema.get(\'description\', \'\'),\n            })\n\n    if hasattr(func, \'cookies\'):\n        cookies = models[func.cookies]\n        for name, schema in cookies[\'properties\'].items():\n            params.append({\n                \'name\': name,\n                \'in\': \'cookie\',\n                \'schema\': schema,\n                \'required\': name in cookies.get(\'required\', []),\n                \'description\': schema.get(\'description\', \'\'),\n            })\n\n    return params\n\n\ndef parse_resp(func):\n    """\n    get the response spec\n\n    If this function does not have explicit ``resp`` but have other models,\n    a ``422 Validation Error`` will be append to the response spec. Since\n    this may be triggered in the validation step.\n    """\n    responses = {}\n    if hasattr(func, \'resp\'):\n        responses = func.resp.generate_spec()\n\n    if \'422\' not in responses and has_model(func):\n        responses[\'422\'] = {\'description\': \'Validation Error\'}\n\n    return responses\n\n\ndef has_model(func):\n    """\n    return True if this function have ``pydantic.BaseModel``\n    """\n    if any(hasattr(func, x) for x in (\'query\', \'json\', \'headers\')):\n        return True\n\n    if hasattr(func, \'resp\') and func.resp.has_model():\n        return True\n\n    return False\n\n\ndef parse_code(http_code):\n    """\n    get the code of this HTTP status\n\n    :param str http_code: format like ``HTTP_200``\n    """\n    match = HTTP_CODE.match(http_code)\n    if not match:\n        return None\n    return match.group(\'code\')\n\n\ndef parse_name(func):\n    """\n    the func can be\n\n        * undecorated functions\n        * decorated functions\n        * decorated class methods\n    """\n    return func.__name__\n\n\ndef default_before_handler(req, resp, req_validation_error, instance):\n    """\n    default handler called before the endpoint function after the request validation\n\n    :param req: request provided by the web framework\n    :param resp: response generated by SpecTree that will be returned\n        if the validation error is not None\n    :param req_validation_error: request validation error\n    :param instance: class instance if the endpoint function is a class method\n    """\n    if req_validation_error:\n        logger.info(\n            \'422 Validation Error\',\n            extra={\n                \'spectree_model\': req_validation_error.model.__name__,\n                \'spectree_validation\': req_validation_error.errors(),\n            },\n        )\n\n\ndef default_after_handler(req, resp, resp_validation_error, instance):\n    """\n    default handler called after the response validation\n\n    :param req: request provided by the web framework\n    :param resp: response from the endpoint function (if there is no validation error)\n        or response validation error\n    :param resp_validation_error: response validation error\n    :param instance: class instance if the endpoint function is a class method\n    """\n    if resp_validation_error:\n        logger.info(\n            \'500 Response Validation Error\',\n            extra={\n                \'spectree_model\': resp_validation_error.model.__name__,\n                \'spectree_validation\': resp_validation_error.errors(),\n            },\n        )\n\n```\nFile Name: setup.py\n\nFile Content:\n ```python\nfrom setuptools import setup, find_packages\nfrom os import path\nfrom io import open\n\n\nhere = path.abspath(path.dirname(__file__))\n\nwith open(path.join(here, \'README.md\'), encoding=\'utf-8\') as f:\n    readme = f.read()\n\nwith open(path.join(here, \'requirements.txt\'), encoding=\'utf-8\') as f:\n    requires = [req.strip() for req in f if req]\n\n\nsetup(\n    name=\'spectree\',\n    version=\'0.3.8\',\n    author=\'Keming Yang\',\n    author_email=\'kemingy94@gmail.com\',\n    description=(\'generate OpenAPI document and validate request&response \'\n                 \'with Python annotations.\'),\n    long_description=readme,\n    long_description_content_type=\'text/markdown\',\n    url=\'https://github.com/0b01001001/spectree\',\n    packages=find_packages(exclude=[\'examples*\', \'tests*\']),\n    package_data={\n    },\n    classifiers=[\n        \'Programming Language :: Python :: 3 :: Only\',\n        \'Programming Language :: Python :: 3.6\',\n        \'Programming Language :: Python :: 3.7\',\n        \'Programming Language :: Python :: 3.8\',\n        \'Operating System :: OS Independent\',\n        \'Topic :: Software Development :: Libraries :: Python Modules\',\n    ],\n    python_requires=\'>=3.6\',\n    install_requires=requires,\n    extras_require={\n        \'flask\': [\'flask\'],\n        \'falcon\': [\'falcon\'],\n        \'starlette\': [\'starlette\', \'requests\'],\n    },\n    zip_safe=False,\n    entry_points={\n        \'console_scripts\': [],\n    },\n)\n\n```\nFile Name: spectree/plugins/starlette_plugin.py\n\nFile Content:\n ```python\nimport inspect\nfrom collections import namedtuple\nfrom functools import partial\nfrom json import JSONDecodeError\nfrom json import loads as json_loads\n\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin, Context\nfrom .page import PAGES\n\nMETHODS = {\'get\', \'post\', \'put\', \'patch\', \'delete\'}\nRoute = namedtuple(\'Route\', [\'path\', \'methods\', \'func\'])\n\n\nclass StarlettePlugin(BasePlugin):\n    def __init__(self, spectree):\n        super().__init__(spectree)\n        from starlette.convertors import CONVERTOR_TYPES\n        self.conv2type = {\n            conv: typ for typ, conv in CONVERTOR_TYPES.items()\n        }\n\n    def register_route(self, app):\n        self.app = app\n        from starlette.responses import JSONResponse, HTMLResponse\n\n        self.app.add_route(\n            self.config.spec_url,\n            lambda request: JSONResponse(self.spectree.spec),\n        )\n\n        for ui in PAGES:\n            self.app.add_route(\n                f\'/{self.config.PATH}/{ui}\',\n                lambda request, ui=ui: HTMLResponse(\n                    PAGES[ui].format(self.config.spec_url)\n                ),\n            )\n\n    async def request_validation(self, request, query, json, headers, cookies):\n        request.context = Context(\n            query.parse_obj(request.query_params) if query else None,\n            json.parse_obj(json_loads(await request.body() or \'{}\')) if json else None,\n            headers.parse_obj(request.headers) if headers else None,\n            cookies.parse_obj(request.cookies) if cookies else None,\n        )\n\n    async def validate(self,\n                       func,\n                       query, json, headers, cookies, resp,\n                       before, after,\n                       *args, **kwargs):\n        from starlette.responses import JSONResponse\n\n        # NOTE: If func is a `HTTPEndpoint`, it should have \'.\' in its ``__qualname__``\n        # This is not elegant. But it seems `inspect` doesn\'t work here.\n        instance = args[0] if \'.\' in func.__qualname__ else None\n        request = args[1] if \'.\' in func.__qualname__ else args[0]\n        response = None\n        req_validation_error, resp_validation_error, json_decode_error = None, None, None\n\n        try:\n            await self.request_validation(request, query, json, headers, cookies)\n        except ValidationError as err:\n            req_validation_error = err\n            response = JSONResponse(err.errors(), 422)\n        except JSONDecodeError as err:\n            json_decode_error = err\n            self.logger.info(\n                \'422 Validation Error\',\n                extra={\'spectree_json_decode_error\': str(err)}\n            )\n            response = JSONResponse({\'error_msg\': str(err)}, 422)\n\n        before(request, response, req_validation_error, instance)\n        if req_validation_error or json_decode_error:\n            return response\n\n        if inspect.iscoroutinefunction(func):\n            response = await func(*args, **kwargs)\n        else:\n            response = func(*args, **kwargs)\n\n        if resp:\n            model = resp.find_model(response.status_code)\n            if model:\n                try:\n                    model.validate(json_loads(response.body))\n                except ValidationError as err:\n                    resp_validation_error = err\n                    response = JSONResponse(err.errors(), 500)\n\n        after(request, response, resp_validation_error, instance)\n\n        return response\n\n    def find_routes(self):\n        routes = []\n\n        def parse_route(app, prefix=\'\'):\n            for route in app.routes:\n                if route.path.startswith(f\'/{self.config.PATH}\'):\n                    continue\n\n                func = route.app\n                if isinstance(func, partial):\n                    try:\n                        func = func.__wrapped__\n                    except AttributeError:\n                        pass\n\n                if inspect.isclass(func):\n                    for method in METHODS:\n                        if getattr(func, method, None):\n                            routes.append(Route(\n                                f\'{prefix}{route.path}\',\n                                {method.upper()},\n                                getattr(func, method)\n                            ))\n                elif inspect.isfunction(func):\n                    routes.append(Route(\n                        f\'{prefix}{route.path}\',\n                        route.methods,\n                        route.endpoint))\n                else:\n                    parse_route(route, prefix=f\'{prefix}{route.path}\')\n\n        parse_route(self.app)\n        return routes\n\n    def bypass(self, func, method):\n        if method in [\'HEAD\', \'OPTIONS\']:\n            return True\n        return False\n\n    def parse_func(self, route):\n        for method in route.methods or [\'GET\']:\n            yield method, route.func\n\n    def parse_path(self, route):\n        from starlette.routing import compile_path\n        _, path, variables = compile_path(route.path)\n        parameters = []\n\n        for name, conv in variables.items():\n            schema = None\n            typ = self.conv2type[conv]\n            if typ == \'int\':\n                schema = {\n                    \'type\': \'integer\',\n                    \'format\': \'int32\'\n                }\n            elif typ == \'float\':\n                schema = {\n                    \'type\': \'number\',\n                    \'format\': \'float\',\n                }\n            elif typ == \'path\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'path\',\n                }\n            elif typ == \'str\':\n                schema = {\'type\': \'string\'}\n\n            parameters.append({\n                \'name\': name,\n                \'in\': \'path\',\n                \'required\': True,\n                \'schema\': schema,\n            })\n\n        return path, parameters\n\n```\nFile Name: spectree/plugins/falcon_plugin.py\n\nFile Content:\n ```python\nimport inspect\nimport re\nfrom functools import partial\n\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin\nfrom .page import PAGES\n\n\nclass OpenAPI:\n    def __init__(self, spec):\n        self.spec = spec\n\n    def on_get(self, req, resp):\n        resp.media = self.spec\n\n\nclass DocPage:\n    def __init__(self, html, spec_url):\n        self.page = html.format(spec_url)\n\n    def on_get(self, req, resp):\n        resp.content_type = \'text/html\'\n        resp.body = self.page\n\n\nDOC_CLASS = [x.__name__ for x in (DocPage, OpenAPI)]\n\n\nclass FalconPlugin(BasePlugin):\n    def __init__(self, spectree):\n        super().__init__(spectree)\n        from falcon.routing.compiled import _FIELD_PATTERN\n\n        self.FIELD_PATTERN = _FIELD_PATTERN\n        # NOTE from `falcon.routing.compiled.CompiledRouterNode`\n        self.ESCAPE = r\'[\\.\\(\\)\\[\\]\\?\\$\\*\\+\\^\\|]\'\n        self.ESCAPE_TO = r\'\\\\\\g<0>\'\n        self.EXTRACT = r\'{\\2}\'\n        # NOTE this regex is copied from werkzeug.routing._converter_args_re and\n        # modified to support only int args\n        self.INT_ARGS = re.compile(r"""\n            ((?P<name>\\w+)\\s*=\\s*)?\n            (?P<value>\\d+)\\s*\n        """, re.VERBOSE)\n        self.INT_ARGS_NAMES = (\'num_digits\', \'min\', \'max\')\n\n    def register_route(self, app):\n        self.app = app\n        self.app.add_route(\n            self.config.spec_url, OpenAPI(self.spectree.spec)\n        )\n        for ui in PAGES:\n            self.app.add_route(\n                f\'/{self.config.PATH}/{ui}\',\n                DocPage(PAGES[ui], self.config.spec_url),\n            )\n\n    def find_routes(self):\n        routes = []\n\n        def find_node(node):\n            if node.resource and node.resource.__class__.__name__ not in DOC_CLASS:\n                routes.append(node)\n\n            for child in node.children:\n                find_node(child)\n\n        for route in self.app._router._roots:\n            find_node(route)\n\n        return routes\n\n    def parse_func(self, route):\n        return route.method_map.items()\n\n    def parse_path(self, route):\n        subs, parameters = [], []\n        for segment in route.uri_template.strip(\'/\').split(\'/\'):\n            matches = self.FIELD_PATTERN.finditer(segment)\n            if not matches:\n                subs.append(segment)\n                continue\n\n            escaped = re.sub(self.ESCAPE, self.ESCAPE_TO, segment)\n            subs.append(self.FIELD_PATTERN.sub(self.EXTRACT, escaped))\n\n            for field in matches:\n                variable, converter, argstr = [field.group(name) for name in\n                                               (\'fname\', \'cname\', \'argstr\')]\n\n                if converter == \'int\':\n                    if argstr is None:\n                        argstr = \'\'\n\n                    arg_values = [None, None, None]\n                    for index, match in enumerate(self.INT_ARGS.finditer(argstr)):\n                        name, value = match.group(\'name\'), match.group(\'value\')\n                        if name:\n                            index = self.INT_ARGS_NAMES.index(name)\n                        arg_values[index] = value\n\n                    num_digits, minumum, maximum = arg_values\n                    schema = {\n                        \'type\': \'integer\',\n                        \'format\': f\'int{num_digits}\' if num_digits else \'int32\',\n                    }\n                    if minumum:\n                        schema[\'minimum\'] = minumum\n                    if maximum:\n                        schema[\'maximum\'] = maximum\n                elif converter == \'uuid\':\n                    schema = {\n                        \'type\': \'string\',\n                        \'format\': \'uuid\'\n                    }\n                elif converter == \'dt\':\n                    schema = {\n                        \'type\': \'string\',\n                        \'format\': \'date-time\',\n                    }\n                else:\n                    # no converter specified or customized converters\n                    schema = {\'type\': \'string\'}\n\n                parameters.append({\n                    \'name\': variable,\n                    \'in\': \'path\',\n                    \'required\': True,\n                    \'schema\': schema,\n                })\n\n        return f\'/{"/".join(subs)}\', parameters\n\n    def request_validation(self, req, query, json, headers, cookies):\n        if query:\n            req.context.query = query.parse_obj(req.params)\n        if headers:\n            req.context.headers = headers.parse_obj(req.headers)\n        if cookies:\n            req.context.cookies = cookies.parse_obj(req.cookies)\n        media = req.media or {}\n        if json:\n            req.context.json = json.parse_obj(media)\n\n    def validate(self,\n                 func,\n                 query, json, headers, cookies, resp,\n                 before, after,\n                 *args, **kwargs):\n        # falcon endpoint method arguments: (self, req, resp)\n        _self, _req, _resp = args[:3]\n        req_validation_error, resp_validation_error = None, None\n        try:\n            self.request_validation(_req, query, json, headers, cookies)\n\n        except ValidationError as err:\n            req_validation_error = err\n            _resp.status = \'422 Unprocessable Entity\'\n            _resp.media = err.errors()\n\n        before(_req, _resp, req_validation_error, _self)\n        if req_validation_error:\n            return\n\n        func(*args, **kwargs)\n        if resp and resp.has_model():\n            model = resp.find_model(_resp.status[:3])\n            if model:\n                try:\n                    model.validate(_resp.media)\n                except ValidationError as err:\n                    resp_validation_error = err\n                    _resp.status = \'500 Internal Service Response Validation Error\'\n                    _resp.media = err.errors()\n\n        after(_req, _resp, resp_validation_error, _self)\n\n    def bypass(self, func, method):\n        if not isinstance(func, partial):\n            return False\n        if inspect.ismethod(func.func):\n            return False\n        # others are <cyfunction>\n        return True\n\n```\nFile Name: spectree/plugins/flask_plugin.py\n\nFile Content:\n ```python\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin, Context\nfrom .page import PAGES\n\n\nclass FlaskPlugin(BasePlugin):\n    blueprint_state = None\n\n    def find_routes(self):\n        from flask import current_app\n        if self.blueprint_state:\n            excludes = [f\'{self.blueprint_state.blueprint.name}.{ep}\'\n                        for ep in [\'static\', \'openapi\'] + [f\'doc_page_{ui}\' for ui in PAGES]]\n            for rule in current_app.url_map.iter_rules():\n                if self.blueprint_state.url_prefix and \\\n                        not str(rule).startswith(self.blueprint_state.url_prefix):\n                    continue\n                if rule.endpoint in excludes:\n                    continue\n                yield rule\n        else:\n            for rule in self.app.url_map.iter_rules():\n                if any(str(rule).startswith(path) for path in (\n                        f\'/{self.config.PATH}\', \'/static\'\n                )):\n                    continue\n                yield rule\n\n    def bypass(self, func, method):\n        if method in [\'HEAD\', \'OPTIONS\']:\n            return True\n        return False\n\n    def parse_func(self, route):\n        if self.blueprint_state:\n            func = self.blueprint_state.app.view_functions[route.endpoint]\n        else:\n            func = self.app.view_functions[route.endpoint]\n\n        for method in route.methods:\n            yield method, func\n\n    def parse_path(self, route):\n        from werkzeug.routing import parse_rule, parse_converter_args\n\n        subs = []\n        parameters = []\n\n        for converter, arguments, variable in parse_rule(str(route)):\n            if converter is None:\n                subs.append(variable)\n                continue\n            subs.append(f\'{{{variable}}}\')\n\n            args, kwargs = [], {}\n\n            if arguments:\n                args, kwargs = parse_converter_args(arguments)\n\n            schema = None\n            if converter == \'any\':\n                schema = {\n                    \'type\': \'array\',\n                    \'items\': {\n                        \'type\': \'string\',\n                        \'enum\': args,\n                    }\n                }\n            elif converter == \'int\':\n                schema = {\n                    \'type\': \'integer\',\n                    \'format\': \'int32\',\n                }\n                if \'max\' in kwargs:\n                    schema[\'maximum\'] = kwargs[\'max\']\n                if \'min\' in kwargs:\n                    schema[\'minimum\'] = kwargs[\'min\']\n            elif converter == \'float\':\n                schema = {\n                    \'type\': \'number\',\n                    \'format\': \'float\',\n                }\n            elif converter == \'uuid\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'uuid\',\n                }\n            elif converter == \'path\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'path\',\n                }\n            elif converter == \'string\':\n                schema = {\n                    \'type\': \'string\',\n                }\n                for prop in [\'length\', \'maxLength\', \'minLength\']:\n                    if prop in kwargs:\n                        schema[prop] = kwargs[prop]\n            elif converter == \'default\':\n                schema = {\'type\': \'string\'}\n\n            parameters.append({\n                \'name\': variable,\n                \'in\': \'path\',\n                \'required\': True,\n                \'schema\': schema,\n            })\n\n        return \'\'.join(subs), parameters\n\n    def request_validation(self, request, query, json, headers, cookies):\n        req_query = request.args or {}\n        req_json = request.get_json() or {}\n        req_headers = request.headers or {}\n        req_cookies = request.cookies or {}\n        request.context = Context(\n            query.parse_obj(req_query) if query else None,\n            json.parse_obj(req_json) if json else None,\n            headers.parse_obj(req_headers) if headers else None,\n            cookies.parse_obj(req_cookies) if cookies else None,\n        )\n\n    def validate(self,\n                 func,\n                 query, json, headers, cookies, resp,\n                 before, after,\n                 *args, **kwargs):\n        from flask import request, abort, make_response, jsonify\n\n        response, req_validation_error, resp_validation_error = None, None, None\n        try:\n            self.request_validation(request, query, json, headers, cookies)\n        except ValidationError as err:\n            req_validation_error = err\n            response = make_response(jsonify(err.errors()), 422)\n\n        before(request, response, req_validation_error, None)\n        if req_validation_error:\n            abort(response)\n\n        response = make_response(func(*args, **kwargs))\n\n        if resp and resp.has_model():\n            model = resp.find_model(response.status_code)\n            if model:\n                try:\n                    model.validate(response.get_json())\n                except ValidationError as err:\n                    resp_validation_error = err\n                    response = make_response(jsonify(\n                        {\'message\': \'response validation error\'}\n                    ), 500)\n\n        after(request, response, resp_validation_error, None)\n\n        return response\n\n    def register_route(self, app):\n        self.app = app\n        from flask import jsonify, Blueprint\n\n        self.app.add_url_rule(\n            self.config.spec_url,\n            \'openapi\',\n            lambda: jsonify(self.spectree.spec),\n        )\n\n        if isinstance(app, Blueprint):\n            def gen_doc_page(ui):\n                spec_url = self.config.spec_url\n                if self.blueprint_state.url_prefix is not None:\n                    spec_url = \'/\'.join((\n                        self.blueprint_state.url_prefix.rstrip(\'/\'),\n                        self.config.spec_url.lstrip(\'/\'))\n                    )\n\n                return PAGES[ui].format(spec_url)\n\n            for ui in PAGES:\n                app.add_url_rule(\n                    f\'/{self.config.PATH}/{ui}\',\n                    f\'doc_page_{ui}\',\n                    lambda ui=ui: gen_doc_page(ui)\n                )\n\n            app.record(lambda state: setattr(self, \'blueprint_state\', state))\n        else:\n            for ui in PAGES:\n                self.app.add_url_rule(\n                    f\'/{self.config.PATH}/{ui}\',\n                    f\'doc_page_{ui}\',\n                    lambda ui=ui: PAGES[ui].format(self.config.spec_url)\n                )\n\n```\nFile Name: spectree/plugins/starlette_plugin.py\n\nFile Content:\n ```python\nimport inspect\nfrom collections import namedtuple\nfrom functools import partial\nfrom json import JSONDecodeError\nfrom json import loads as json_loads\n\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin, Context\nfrom .page import PAGES\n\nMETHODS = {\'get\', \'post\', \'put\', \'patch\', \'delete\'}\nRoute = namedtuple(\'Route\', [\'path\', \'methods\', \'func\'])\n\n\nclass StarlettePlugin(BasePlugin):\n    def __init__(self, spectree):\n        super().__init__(spectree)\n        from starlette.convertors import CONVERTOR_TYPES\n        self.conv2type = {\n            conv: typ for typ, conv in CONVERTOR_TYPES.items()\n        }\n\n    def register_route(self, app):\n        self.app = app\n        from starlette.responses import JSONResponse, HTMLResponse\n\n        self.app.add_route(\n            self.config.spec_url,\n            lambda request: JSONResponse(self.spectree.spec),\n        )\n\n        for ui in PAGES:\n            self.app.add_route(\n                f\'/{self.config.PATH}/{ui}\',\n                lambda request, ui=ui: HTMLResponse(\n                    PAGES[ui].format(self.config.spec_url)\n                ),\n            )\n\n    async def request_validation(self, request, query, json, headers, cookies):\n        request.context = Context(\n            query.parse_obj(request.query_params) if query else None,\n            json.parse_obj(json_loads(await request.body() or \'{}\')) if json else None,\n            headers.parse_obj(request.headers) if headers else None,\n            cookies.parse_obj(request.cookies) if cookies else None,\n        )\n\n    async def validate(self,\n                       func,\n                       query, json, headers, cookies, resp,\n                       before, after,\n                       *args, **kwargs):\n        from starlette.responses import JSONResponse\n\n        # NOTE: If func is a `HTTPEndpoint`, it should have \'.\' in its ``__qualname__``\n        # This is not elegant. But it seems `inspect` doesn\'t work here.\n        instance = args[0] if \'.\' in func.__qualname__ else None\n        request = args[1] if \'.\' in func.__qualname__ else args[0]\n        response = None\n        req_validation_error, resp_validation_error, json_decode_error = None, None, None\n\n        try:\n            await self.request_validation(request, query, json, headers, cookies)\n        except ValidationError as err:\n            req_validation_error = err\n            response = JSONResponse(err.errors(), 422)\n        except JSONDecodeError as err:\n            json_decode_error = err\n            self.logger.info(\n                \'422 Validation Error\',\n                extra={\'spectree_json_decode_error\': str(err)}\n            )\n            response = JSONResponse({\'error_msg\': str(err)}, 422)\n\n        before(request, response, req_validation_error, instance)\n        if req_validation_error or json_decode_error:\n            return response\n\n        if inspect.iscoroutinefunction(func):\n            response = await func(*args, **kwargs)\n        else:\n            response = func(*args, **kwargs)\n\n        if resp:\n            model = resp.find_model(response.status_code)\n            if model:\n                try:\n                    model.validate(json_loads(response.body))\n                except ValidationError as err:\n                    resp_validation_error = err\n                    response = JSONResponse(err.errors(), 500)\n\n        after(request, response, resp_validation_error, instance)\n\n        return response\n\n    def find_routes(self):\n        routes = []\n\n        def parse_route(app, prefix=\'\'):\n            for route in app.routes:\n                if route.path.startswith(f\'/{self.config.PATH}\'):\n                    continue\n\n                func = route.app\n                if isinstance(func, partial):\n                    try:\n                        func = func.__wrapped__\n                    except AttributeError:\n                        pass\n\n                if inspect.isclass(func):\n                    for method in METHODS:\n                        if getattr(func, method, None):\n                            routes.append(Route(\n                                f\'{prefix}{route.path}\',\n                                {method.upper()},\n                                getattr(func, method)\n                            ))\n                elif inspect.isfunction(func):\n                    routes.append(Route(\n                        f\'{prefix}{route.path}\',\n                        route.methods,\n                        route.endpoint))\n                else:\n                    parse_route(route, prefix=f\'{prefix}{route.path}\')\n\n        parse_route(self.app)\n        return routes\n\n    def bypass(self, func, method):\n        if method in [\'HEAD\', \'OPTIONS\']:\n            return True\n        return False\n\n    def parse_func(self, route):\n        for method in route.methods or [\'GET\']:\n            yield method, route.func\n\n    def parse_path(self, route):\n        from starlette.routing import compile_path\n        _, path, variables = compile_path(route.path)\n        parameters = []\n\n        for name, conv in variables.items():\n            schema = None\n            typ = self.conv2type[conv]\n            if typ == \'int\':\n                schema = {\n                    \'type\': \'integer\',\n                    \'format\': \'int32\'\n                }\n            elif typ == \'float\':\n                schema = {\n                    \'type\': \'number\',\n                    \'format\': \'float\',\n                }\n            elif typ == \'path\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'path\',\n                }\n            elif typ == \'str\':\n                schema = {\'type\': \'string\'}\n\n            parameters.append({\n                \'name\': name,\n                \'in\': \'path\',\n                \'required\': True,\n                \'schema\': schema,\n            })\n\n        return path, parameters\n\n```\nFile Name: spectree/plugins/falcon_plugin.py\n\nFile Content:\n ```python\nimport inspect\nimport re\nfrom functools import partial\n\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin\nfrom .page import PAGES\n\n\nclass OpenAPI:\n    def __init__(self, spec):\n        self.spec = spec\n\n    def on_get(self, req, resp):\n        resp.media = self.spec\n\n\nclass DocPage:\n    def __init__(self, html, spec_url):\n        self.page = html.format(spec_url)\n\n    def on_get(self, req, resp):\n        resp.content_type = \'text/html\'\n        resp.body = self.page\n\n\nDOC_CLASS = [x.__name__ for x in (DocPage, OpenAPI)]\n\n\nclass FalconPlugin(BasePlugin):\n    def __init__(self, spectree):\n        super().__init__(spectree)\n        from falcon.routing.compiled import _FIELD_PATTERN\n\n        self.FIELD_PATTERN = _FIELD_PATTERN\n        # NOTE from `falcon.routing.compiled.CompiledRouterNode`\n        self.ESCAPE = r\'[\\.\\(\\)\\[\\]\\?\\$\\*\\+\\^\\|]\'\n        self.ESCAPE_TO = r\'\\\\\\g<0>\'\n        self.EXTRACT = r\'{\\2}\'\n        # NOTE this regex is copied from werkzeug.routing._converter_args_re and\n        # modified to support only int args\n        self.INT_ARGS = re.compile(r"""\n            ((?P<name>\\w+)\\s*=\\s*)?\n            (?P<value>\\d+)\\s*\n        """, re.VERBOSE)\n        self.INT_ARGS_NAMES = (\'num_digits\', \'min\', \'max\')\n\n    def register_route(self, app):\n        self.app = app\n        self.app.add_route(\n            self.config.spec_url, OpenAPI(self.spectree.spec)\n        )\n        for ui in PAGES:\n            self.app.add_route(\n                f\'/{self.config.PATH}/{ui}\',\n                DocPage(PAGES[ui], self.config.spec_url),\n            )\n\n    def find_routes(self):\n        routes = []\n\n        def find_node(node):\n            if node.resource and node.resource.__class__.__name__ not in DOC_CLASS:\n                routes.append(node)\n\n            for child in node.children:\n                find_node(child)\n\n        for route in self.app._router._roots:\n            find_node(route)\n\n        return routes\n\n    def parse_func(self, route):\n        return route.method_map.items()\n\n    def parse_path(self, route):\n        subs, parameters = [], []\n        for segment in route.uri_template.strip(\'/\').split(\'/\'):\n            matches = self.FIELD_PATTERN.finditer(segment)\n            if not matches:\n                subs.append(segment)\n                continue\n\n            escaped = re.sub(self.ESCAPE, self.ESCAPE_TO, segment)\n            subs.append(self.FIELD_PATTERN.sub(self.EXTRACT, escaped))\n\n            for field in matches:\n                variable, converter, argstr = [field.group(name) for name in\n                                               (\'fname\', \'cname\', \'argstr\')]\n\n                if converter == \'int\':\n                    if argstr is None:\n                        argstr = \'\'\n\n                    arg_values = [None, None, None]\n                    for index, match in enumerate(self.INT_ARGS.finditer(argstr)):\n                        name, value = match.group(\'name\'), match.group(\'value\')\n                        if name:\n                            index = self.INT_ARGS_NAMES.index(name)\n                        arg_values[index] = value\n\n                    num_digits, minumum, maximum = arg_values\n                    schema = {\n                        \'type\': \'integer\',\n                        \'format\': f\'int{num_digits}\' if num_digits else \'int32\',\n                    }\n                    if minumum:\n                        schema[\'minimum\'] = minumum\n                    if maximum:\n                        schema[\'maximum\'] = maximum\n                elif converter == \'uuid\':\n                    schema = {\n                        \'type\': \'string\',\n                        \'format\': \'uuid\'\n                    }\n                elif converter == \'dt\':\n                    schema = {\n                        \'type\': \'string\',\n                        \'format\': \'date-time\',\n                    }\n                else:\n                    # no converter specified or customized converters\n                    schema = {\'type\': \'string\'}\n\n                parameters.append({\n                    \'name\': variable,\n                    \'in\': \'path\',\n                    \'required\': True,\n                    \'schema\': schema,\n                })\n\n        return f\'/{"/".join(subs)}\', parameters\n\n    def request_validation(self, req, query, json, headers, cookies):\n        if query:\n            req.context.query = query.parse_obj(req.params)\n        if headers:\n            req.context.headers = headers.parse_obj(req.headers)\n        if cookies:\n            req.context.cookies = cookies.parse_obj(req.cookies)\n        media = req.media or {}\n        if json:\n            req.context.json = json.parse_obj(media)\n\n    def validate(self,\n                 func,\n                 query, json, headers, cookies, resp,\n                 before, after,\n                 *args, **kwargs):\n        # falcon endpoint method arguments: (self, req, resp)\n        _self, _req, _resp = args[:3]\n        req_validation_error, resp_validation_error = None, None\n        try:\n            self.request_validation(_req, query, json, headers, cookies)\n\n        except ValidationError as err:\n            req_validation_error = err\n            _resp.status = \'422 Unprocessable Entity\'\n            _resp.media = err.errors()\n\n        before(_req, _resp, req_validation_error, _self)\n        if req_validation_error:\n            return\n\n        func(*args, **kwargs)\n        if resp and resp.has_model():\n            model = resp.find_model(_resp.status[:3])\n            if model:\n                try:\n                    model.validate(_resp.media)\n                except ValidationError as err:\n                    resp_validation_error = err\n                    _resp.status = \'500 Internal Service Response Validation Error\'\n                    _resp.media = err.errors()\n\n        after(_req, _resp, resp_validation_error, _self)\n\n    def bypass(self, func, method):\n        if not isinstance(func, partial):\n            return False\n        if inspect.ismethod(func.func):\n            return False\n        # others are <cyfunction>\n        return True\n\n```\nFile Name: spectree/plugins/flask_plugin.py\n\nFile Content:\n ```python\nfrom pydantic import ValidationError\n\nfrom .base import BasePlugin, Context\nfrom .page import PAGES\n\n\nclass FlaskPlugin(BasePlugin):\n    blueprint_state = None\n\n    def find_routes(self):\n        from flask import current_app\n        if self.blueprint_state:\n            excludes = [f\'{self.blueprint_state.blueprint.name}.{ep}\'\n                        for ep in [\'static\', \'openapi\'] + [f\'doc_page_{ui}\' for ui in PAGES]]\n            for rule in current_app.url_map.iter_rules():\n                if self.blueprint_state.url_prefix and \\\n                        not str(rule).startswith(self.blueprint_state.url_prefix):\n                    continue\n                if rule.endpoint in excludes:\n                    continue\n                yield rule\n        else:\n            for rule in self.app.url_map.iter_rules():\n                if any(str(rule).startswith(path) for path in (\n                        f\'/{self.config.PATH}\', \'/static\'\n                )):\n                    continue\n                yield rule\n\n    def bypass(self, func, method):\n        if method in [\'HEAD\', \'OPTIONS\']:\n            return True\n        return False\n\n    def parse_func(self, route):\n        if self.blueprint_state:\n            func = self.blueprint_state.app.view_functions[route.endpoint]\n        else:\n            func = self.app.view_functions[route.endpoint]\n\n        for method in route.methods:\n            yield method, func\n\n    def parse_path(self, route):\n        from werkzeug.routing import parse_rule, parse_converter_args\n\n        subs = []\n        parameters = []\n\n        for converter, arguments, variable in parse_rule(str(route)):\n            if converter is None:\n                subs.append(variable)\n                continue\n            subs.append(f\'{{{variable}}}\')\n\n            args, kwargs = [], {}\n\n            if arguments:\n                args, kwargs = parse_converter_args(arguments)\n\n            schema = None\n            if converter == \'any\':\n                schema = {\n                    \'type\': \'array\',\n                    \'items\': {\n                        \'type\': \'string\',\n                        \'enum\': args,\n                    }\n                }\n            elif converter == \'int\':\n                schema = {\n                    \'type\': \'integer\',\n                    \'format\': \'int32\',\n                }\n                if \'max\' in kwargs:\n                    schema[\'maximum\'] = kwargs[\'max\']\n                if \'min\' in kwargs:\n                    schema[\'minimum\'] = kwargs[\'min\']\n            elif converter == \'float\':\n                schema = {\n                    \'type\': \'number\',\n                    \'format\': \'float\',\n                }\n            elif converter == \'uuid\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'uuid\',\n                }\n            elif converter == \'path\':\n                schema = {\n                    \'type\': \'string\',\n                    \'format\': \'path\',\n                }\n            elif converter == \'string\':\n                schema = {\n                    \'type\': \'string\',\n                }\n                for prop in [\'length\', \'maxLength\', \'minLength\']:\n                    if prop in kwargs:\n                        schema[prop] = kwargs[prop]\n            elif converter == \'default\':\n                schema = {\'type\': \'string\'}\n\n            parameters.append({\n                \'name\': variable,\n                \'in\': \'path\',\n                \'required\': True,\n                \'schema\': schema,\n            })\n\n        return \'\'.join(subs), parameters\n\n    def request_validation(self, request, query, json, headers, cookies):\n        req_query = request.args or {}\n        req_json = request.get_json() or {}\n        req_headers = request.headers or {}\n        req_cookies = request.cookies or {}\n        request.context = Context(\n            query.parse_obj(req_query) if query else None,\n            json.parse_obj(req_json) if json else None,\n            headers.parse_obj(req_headers) if headers else None,\n            cookies.parse_obj(req_cookies) if cookies else None,\n        )\n\n    def validate(self,\n                 func,\n                 query, json, headers, cookies, resp,\n                 before, after,\n                 *args, **kwargs):\n        from flask import request, abort, make_response, jsonify\n\n        response, req_validation_error, resp_validation_error = None, None, None\n        try:\n            self.request_validation(request, query, json, headers, cookies)\n        except ValidationError as err:\n            req_validation_error = err\n            response = make_response(jsonify(err.errors()), 422)\n\n        before(request, response, req_validation_error, None)\n        if req_validation_error:\n            abort(response)\n\n        response = make_response(func(*args, **kwargs))\n\n        if resp and resp.has_model():\n            model = resp.find_model(response.status_code)\n            if model:\n                try:\n                    model.validate(response.get_json())\n                except ValidationError as err:\n                    resp_validation_error = err\n                    response = make_response(jsonify(\n                        {\'message\': \'response validation error\'}\n                    ), 500)\n\n        after(request, response, resp_validation_error, None)\n\n        return response\n\n    def register_route(self, app):\n        self.app = app\n        from flask import jsonify, Blueprint\n\n        self.app.add_url_rule(\n            self.config.spec_url,\n            \'openapi\',\n            lambda: jsonify(self.spectree.spec),\n        )\n\n        if isinstance(app, Blueprint):\n            def gen_doc_page(ui):\n                spec_url = self.config.spec_url\n                if self.blueprint_state.url_prefix is not None:\n                    spec_url = \'/\'.join((\n                        self.blueprint_state.url_prefix.rstrip(\'/\'),\n                        self.config.spec_url.lstrip(\'/\'))\n                    )\n\n                return PAGES[ui].format(spec_url)\n\n            for ui in PAGES:\n                app.add_url_rule(\n                    f\'/{self.config.PATH}/{ui}\',\n                    f\'doc_page_{ui}\',\n                    lambda ui=ui: gen_doc_page(ui)\n                )\n\n            app.record(lambda state: setattr(self, \'blueprint_state\', state))\n        else:\n            for ui in PAGES:\n                self.app.add_url_rule(\n                    f\'/{self.config.PATH}/{ui}\',\n                    f\'doc_page_{ui}\',\n                    lambda ui=ui: PAGES[ui].format(self.config.spec_url)\n                )\n\n```\n\n\n## Task:\nBased on the bug patterns shown in the examples above, generate a NEW bug and corresponding unittest that:\n1. **MUST be introduced into the Target Code above** (not the example codes)\n2. Is different from all three examples but follows similar realistic programming error patterns\n3. Would be something a developer might accidentally introduce\n4. Maintains the overall structure and functionality of the original code\n5. **Should be DIFFICULT and SUBTLE** - requiring deep understanding of the code logic, careful debugging skills, and thorough analysis to identify and fix\n6. Should not cause obvious syntax errors or immediate crashes, but rather introduce logical errors that manifest under specific conditions\n7. **Must be accompanied by a comprehensive unittest** that thoroughly tests the functionality and can detect the introduced bug\n\n## Output Requirements:\n- Provide COMPLETE buggy code for each modified file (not partial snippets)\n- Provide a COMPLETE unittest file that tests the functionality and detects the bug\n- The unittest should pass when testing the original clean code but fail when testing the buggy code\n- Follow the exact output format below for easy parsing\n- **GENERATE EXTENSIVE TEST CASES**: Create at least 8-12 individual test methods to ensure comprehensive coverage\n\n## Output Format:\n===PROBLEM_STATEMENT_START===\n[Describe the bug as a GitHub issue in the same format as the three examples above - focus on what\'s wrong, the symptoms, and impact. Keep it concise and user-focused without technical implementation details.]\n===PROBLEM_STATEMENT_END===\n\n===BUG_ANALYSIS_START===\n**Test Detection Analysis:**\n- Which specific test methods/cases in the generated unittest will detect this bug\n- Why the bug causes those particular tests to fail\n- Under what conditions the bug manifests\n- Expected vs actual behavior that causes test failures\n\n**Bug Characteristics:**\n- Type of programming error introduced\n- Why this bug is subtle and difficult to detect through code review\n- What makes this bug realistic (common developer mistakes)\n\n**Unittest Design:**\n- Key test scenarios covered by the unittest\n- How the unittest comprehensively validates the functionality\n- Specific test cases designed to catch the introduced bug\n- Description of pass-to-pass test cases that verify core functionality remains intact\n===BUG_ANALYSIS_END===\n\n===BUGGY_FILES_START===\n===FILE_START===\nFILE_PATH: [exact file path]\n===CODE_START===\n```python\n[complete buggy code for this file - must be the full file content, not snippets]\n```\n===CODE_END===\n===FILE_END===\n\n===FILE_START===\nFILE_PATH: [exact file path for second file if needed]\n===CODE_START===\n```python\n[complete buggy code for this file - must be the full file content, not snippets]\n```\n===CODE_END===\n===FILE_END===\n\n[Repeat FILE_START/FILE_END blocks for additional files if necessary]\n===BUGGY_FILES_END===\n\n===UNITTEST_FILE_START===\nFILE_PATH: [unittest file path, e.g., test_[module_name].py]\n===CODE_START===\n```python\n[complete unittest code that thoroughly tests the functionality and can detect the introduced bug]\n```\n===CODE_END===\n===UNITTEST_FILE_END===\n\n## Critical Requirements:\n1. **Bug Difficulty**: The bug must be HARD to detect and fix, involving:\n   - Subtle logical errors that only manifest under specific conditions\n   - Edge cases that are easily overlooked\n   - Complex interactions between different parts of the code\n   - Issues that require deep understanding of the algorithm or data flow\n\n2. **Unittest Quality**: The generated unittest must:\n   - **PASS ALL TESTS** when testing the original clean code\n   - **FAIL SPECIFIC TESTS** when testing the buggy code\n   - **GENERATE AT LEAST 8-12 TEST METHODS** for comprehensive coverage\n   - Include extensive edge cases and boundary conditions\n   - Use appropriate test methods (assertEqual, assertTrue, assertRaises, etc.)\n   - Follow proper unittest structure with setUp/tearDown if needed\n   - Include descriptive test method names and docstrings\n   - Test both normal functionality and error conditions\n   - **Include pass-to-pass test cases** that verify core functionality works correctly in both buggy and clean versions\n\n3. **Problem Statement Style**: Write the problem statement like a GitHub issue:\n   - Focus on user-visible symptoms and impact\n   - Describe what\'s broken from a user perspective\n   - Avoid implementation details or technical jargon\n   - Be concise but clear about the problem\n\n4. **Test Design Strategy**: The unittest should:\n   - **Create COMPREHENSIVE test coverage** with multiple test methods\n   - Include specific test cases that will catch the introduced bug (fail-to-pass cases)\n   - Include test cases that should pass in both versions (pass-to-pass cases)\n   - Test edge cases and boundary conditions extensively\n   - Validate both successful operations and error handling\n   - Use meaningful assertions that clearly indicate what went wrong\n   - **Target minimum 8-12 individual test methods** covering different aspects of functionality\n\n5. **Test Case Distribution**:\n   - **Bug-detecting tests (fail-to-pass)**: 3-5 test methods that fail with the bug but pass with clean code\n   - **Core functionality tests (pass-to-pass)**: 5-9 test methods that should pass regardless of the bug to ensure other functionality is preserved\n   - Include tests for normal use cases, edge cases, error conditions, and integration scenarios\n\n## Important Notes:\n- All code must be wrapped in ```python blocks for proper syntax highlighting\n- Follow the exact delimiters (===SECTION_START===, ===SECTION_END===) for parsing\n- The bug must be introduced to the Target Code, not the example codes\n- Focus on creating sophisticated bugs that challenge debugging skills rather than obvious errors\n- The unittest should be production-quality with proper structure, documentation, and coverage\n- Ensure the unittest is self-contained and can be run independently\n- **EMPHASIZE QUANTITY**: Generate extensive test methods to ensure comprehensive validation\n- The bug should be subtle enough that it might pass casual inspection but will be caught by thorough testing\n- Include both specific bug-detection tests and general functionality preservation tests'
    response = get_deepseek_response(inputs)
    print(response)