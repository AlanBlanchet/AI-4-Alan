from click import command


@command("serve")
def main():
    import subprocess

    subprocess.run(["mkdocs", "build"])
