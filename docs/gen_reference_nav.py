import griffe
import mkdocs_gen_files

# 1) Load your "ai" package from src/
module = griffe.load("ai", search_paths=["src"])

# 2) Create a Nav() object to track how we'll arrange pages
nav = mkdocs_gen_files.Nav()


def walk_module(mod, parent_path=None):
    """Recursively iterate over submodules in the Griffe AST."""
    parent_path = parent_path or []
    current_path = parent_path + [mod.name]
    modules = mod.modules.items()

    if len(modules) == 0:
        yield (current_path, mod)
    else:
        for sub_name, sub_mod in modules:
            yield from walk_module(sub_mod, current_path)


# 3) Generate one .md file per submodule, add them to the Nav
for path_list, mod_obj in walk_module(module):
    # e.g. path_list might be ["ai"], ["ai", "submodule_a"], etc.
    file_path = "reference/" + "/".join(path_list) + ".md"

    print("Generating", file_path)
    # Add this path to the Nav
    nav[path_list] = file_path

    with mkdocs_gen_files.open(file_path, "w") as f:
        f.write(f"# {'.'.join(path_list)}\n\n")
        f.write(f"::: {'.'.join(path_list)}\n")

# 4) Build a .pages file in "reference/" so mkdocs-literate-nav can parse it
nav_content = nav.build_literate_nav()
if not isinstance(nav_content, str):
    nav_content = "".join(nav_content)

# Write the nav structure to docs/reference/.pages
with mkdocs_gen_files.open("reference/.pages", "w") as nav_file:
    nav_file.write(nav_content)
