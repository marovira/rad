import pathlib

if __name__ == "__main__":
    script_path = pathlib.Path(__file__)
    repo_dir = script_path.parent.parent.parent.parent
    deps_path = repo_dir / "dependencies.txt"
    home = pathlib.Path.home()

    with deps_path.open("r", encoding="utf-8") as infile:
        lines = infile.readlines()
        for line in lines:
            line = line.strip()
            if "#" in line:
                continue
            elems = line.split("==")
            name = elems[0].lower()
            version = elems[1]

            out_file = home / f"{name}.txt"
            with out_file.open("w+", encoding="utf-8") as outfile:
                outfile.write(version)
