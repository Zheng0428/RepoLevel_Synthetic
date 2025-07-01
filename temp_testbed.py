import os  
import shutil  
import tempfile  

class TempTestbed:  
    """  
    A context manager that creates a temporary testbed mirroring a source testbed.  
    Files are symlinked by default, with specified files being copied instead.  
    The temporary testbed is automatically cleaned up when the context exits.  
    
    Usage:  
        with TempTestbed(source_dir, copy_files=['config/settings.ini']) as testbed:  
            # Use testbed.path to access the temporary testbed directory  
            modify_file(os.path.join(testbed.path, 'config/settings.ini'))  
            # Use testbed.pytest_path for pytest specific files  
    """  
    
    def __init__(self, source_testbed, copy_files=None, temp_dir=None, pytest_dir=None):  
        """  
        Initialize the temporary testbed.  
        
        Args:  
            source_testbed (str): Path to source testbed directory  
            copy_files (list, optional): List of relative file paths to copy instead of link  
            temp_dir (str, optional): Base directory for the temporary testbed.  
                                      If None, a system temporary directory is used.  
            pytest_dir (str, optional): Base directory for pytest files.  
                                      If None, a system temporary directory is used.  
        """  
        self.source_testbed = os.path.abspath(source_testbed)  
        self.copy_files = copy_files or []  
        self.temp_base = temp_dir  
        self.pytest_base = pytest_dir  
        self.temp_dir = None  
        self.temp_pytest = None  
        self._temp_dir_obj = None      # To hold the TemporaryDirectory object for testbed  
        self._pytest_dir_obj = None    # To hold the TemporaryDirectory object for pytest  
        
    def __enter__(self):  
        """Set up the temporary testbed when entering the context."""  
        # Create a temporary directory to contain our testbed  
        if self.temp_base:  
            os.makedirs(self.temp_base, exist_ok=True)  
            self.temp_dir = tempfile.mkdtemp(dir=self.temp_base)  
        else:  
            self._temp_dir_obj = tempfile.TemporaryDirectory()  
            self.temp_dir = self._temp_dir_obj.name  
        
        # Create a temporary directory for pytest  
        if self.pytest_base:  
            os.makedirs(self.pytest_base, exist_ok=True)  
            self.temp_pytest = tempfile.mkdtemp(dir=self.pytest_base)  
        else:  
            self._pytest_dir_obj = tempfile.TemporaryDirectory()  
            self.temp_pytest = self._pytest_dir_obj.name  
            
        # Prepare the testbed  
        self._prepare_testbed()  
        return self  
    
    def __exit__(self, exc_type, exc_val, exc_tb):  
        """Clean up the temporary testbed when exiting the context."""  
        # Clean up testbed directory  
        if self._temp_dir_obj:  
            self._temp_dir_obj.cleanup()  
        elif self.temp_dir and os.path.exists(self.temp_dir):  
            shutil.rmtree(self.temp_dir,symlinks=False)
        
        # Clean up pytest directory  
        if self._pytest_dir_obj:  
            self._pytest_dir_obj.cleanup()  
        elif self.temp_pytest and os.path.exists(self.temp_pytest):  
            shutil.rmtree(self.temp_pytest)
            
        # Clear references  
        self.temp_dir = None  
        self.temp_pytest = None  
        self._temp_dir_obj = None  
        self._pytest_dir_obj = None  
        
    def _prepare_testbed(self):  
        """  
        Creates a temporary testbed that mirrors the source testbed structure.  
        - All directories are created directly  
        - Files in copy_files are copied  
        - All other files are symbolically linked  
        """  
        # Convert copy_files to a set for faster lookups  
        copy_files_set = set(self.copy_files)  
        
        # Recursively process all items in source_testbed  
        for root, dirs, files in os.walk(self.source_testbed):  
            # Get the relative path from source_testbed  
            rel_path = os.path.relpath(root, self.source_testbed)  
            
            # Create the corresponding directory in temp_testbed  
            if rel_path == '.':  
                target_dir = self.temp_dir  
            else:  
                target_dir = os.path.join(self.temp_dir, rel_path)  
                os.makedirs(target_dir, exist_ok=True)  
            
            # Process files in the current directory  
            for file in files:  
                # Get the source and target file paths  
                source_file = os.path.join(root, file)  
                target_file = os.path.join(target_dir, file)  
                
                # Determine the relative path for checking against copy_files  
                if rel_path == '.':  
                    file_rel_path = file  
                else:  
                    file_rel_path = os.path.join(rel_path, file)  
                
                # Copy or symlink based on whether it's in copy_files  
                if file_rel_path in copy_files_set:  
                    # Copy the file  
                    shutil.copy2(source_file, target_file)  
                else:  
                    # Create a symbolic link  
                    if os.path.exists(target_file):  
                        if os.path.islink(target_file) or os.path.isfile(target_file):  
                            os.remove(target_file)  
                    os.symlink(source_file, target_file)  
    
    @property  
    def path(self):  
        """Return the path to the temporary testbed."""  
        if not self.temp_dir:  
            raise RuntimeError("Temporary testbed is not active. Use within a 'with' statement.")  
        return self.temp_dir  
    
    @property  
    def pytest_path(self):  
        """Return the path to the temporary pytest directory."""  
        if not self.temp_pytest:  
            raise RuntimeError("Temporary pytest directory is not active. Use within a 'with' statement.")  
        return self.temp_pytest  
    
    def print_structure(self, directory=None, prefix="", is_pytest=False):  
        """  
        Print the directory structure of the temporary testbed,  
        indicating which files are symlinks and which are regular copies.  
        
        Args:  
            directory (str, optional): Start directory. If None, uses self.temp_dir or self.temp_pytest  
            prefix (str, optional): Prefix for indentation  
            is_pytest (bool, optional): Whether to print pytest directory structure  
        """  
        if not self.temp_dir and not self.temp_pytest:  
            raise RuntimeError("Temporary testbed is not active. Use within a 'with' statement.")  
            
        if directory is None:  
            if is_pytest:  
                directory = self.temp_pytest  
                print(f"Pytest Directory Structure: {directory}")  
            else:  
                directory = self.temp_dir  
                print(f"Testbed Directory Structure: {directory}")  
                
        # Get items in the current directory  
        items = sorted(os.listdir(directory))  
        
        # Process each item  
        for i, item in enumerate(items):  
            # Determine if this is the last item at this level  
            is_last = i == len(items) - 1  
            
            # Create appropriate prefix for this item  
            if is_last:  
                item_prefix = prefix + "└── "  
                next_prefix = prefix + "    "  
            else:  
                item_prefix = prefix + "├── "  
                next_prefix = prefix + "│   "  
                
            # Get full path  
            item_path = os.path.join(directory, item)  
            
            # Check if it's a directory  
            if os.path.isdir(item_path):  
                print(f"{item_prefix}{item}/")  
                # Recursively print subdirectory  
                self.print_structure(item_path, next_prefix, is_pytest)  
            elif os.path.islink(item_path):  
                # It's a symlink, indicate where it points to  
                link_target = os.readlink(item_path)  
                print(f"{item_prefix}{item} -> {link_target} (symlink)")  
            else:  
                # Regular file, indicate if it was copied  
                if any(item_path.endswith(path) for path in self.copy_files):  
                    print(f"{item_prefix}{item} (copied)")  
                else:  
                    print(f"{item_prefix}{item}")  
    
    def print_all_structures(self):  
        """Print both the testbed and pytest directory structures."""  
        self.print_structure()  # Print testbed structure  
        print("\n")  # Add some spacing  
        self.print_structure(is_pytest=True)  # Print pytest structure  

import re

def get_all_filenames(diff: str) -> dict:
    """从 git diff 输出中提取所有变更的文件名
    
    Args:
        diff: git diff 命令的输出文本
        
    Returns:
        dict: 包含添加、修改和删除的文件列表
    """
    modified_files = re.findall(r"diff --git a/(.*?) b/", diff)

    
    added_files = re.findall(r"--- /dev/null\n\+\+\+ b/(.*?)\n@@", diff)

    removed_files = re.findall(r"--- a/(.*?)\n\+\+\+ /dev/null\n@@",diff)

    modified_files = list(set(modified_files) - set(added_files)-set(removed_files))

    return {
        "added": added_files if added_files else [],
        "modified": modified_files if modified_files else [],
        "removed": removed_files if removed_files else [],
    }

# 测试代码  
if __name__ == "__main__":  
    test_path = "/opt/tiger/expr/testbed/0b01001001__spectree__a091fa/"
    patch = "/mnt/bn/tiktok-mm-5/aiic/users/yiming/data/swe-bench-extra/0b01001001__spectree-64/only_modified_py_patch.diff"
    with open(patch, "r") as f:
        diff = f.read()
    filenames = get_all_filenames(diff)["modified"]
    print(filenames)
    with TempTestbed(test_path, copy_files=filenames) as testbed: 
        temp_dir = testbed.path 
        # Print the directory structure of the temporary testbed  
        testbed.print_structure()  
        # Print the directory structure of the pytest directory  
        testbed.print_structure(is_pytest=True)  
        import pdb;pdb.set_trace()
        # Print all structures in one go  
        testbed.print_all_structures()
    print(temp_dir)

