# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

AGENTLESS_REPAIR = """We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs.

--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap each *SEARCH/REPLACE* edit in a code block as shown in the example above. If you have multiple *SEARCH/REPLACE* edits, use a separate code block for each one."""

CODE_FILE = """
### {path}
{content}
""".strip()

THINKING_SYSTEM_OLD = """
A user will ask you to solve a task. You should first draft your thinking process (inner monologue). Then, generate the solution.

Your response format must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct solution.
</think>
<solution>
Final solution presented to the user.
</solution>
""".strip()

THINKING_SYSTEM = """
You are a helpful programming assistant.
The user will ask you to solve a task. You need to carefully think through and analyze each programming task before providing your answer. Take time to understand the problem requirements, consider edge cases, and ensure your solution is correct and efficient.
""".strip()

LOCALIZATION = """
Please look through a given GitHub issue and repository structure and provide a list of files that one would need to edit or look at to solve the issue.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}


###

Only provide the full path and return at most {n} files. The returned files should be separated by new lines ordered by most to least important and wrapped with ```. For example:

```
most/important/file1.xx
less/important/file2.yy
least/important/file3.zz
```
""".strip()
