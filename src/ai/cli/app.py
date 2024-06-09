import subprocess

from click import Choice, command, option


@command("app", help="Start the app")
@option(
    "-o",
    "--only",
    type=Choice(["backend", "frontend"]),
    default=None,
    help="Start one of backend or frontend only",
)
def main(only: str):
    wait_for = None

    try:
        frontend = None
        backend = None

        if only == "frontend" or only is None:
            frontend = subprocess.Popen(["npm run dev"], shell=True)
            wait_for = frontend

        if only == "backend" or only is None:
            backend = subprocess.Popen(
                ["uvicorn back.main:app"], shell=True, stdout=subprocess.PIPE
            )
            wait_for = backend

        txt = (
            "frontend"
            if only == "frontend"
            else ("backend" if only == "backend" else "frontend and backend")
        )
        print(f"Opening {txt}")
        wait_for.wait()
    finally:
        if frontend is not None:
            frontend.kill()
        if backend is not None:
            backend.kill()
