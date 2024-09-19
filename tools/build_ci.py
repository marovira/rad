import argparse
import dataclasses as dc
import os
import pathlib
import platform
import shutil
import typing

import gdown
from build_dependencies import (
    SDKInfo,
    build_from_script,
    clone_repo,
    rmtree_error,
)


class GDriveURL(typing.TypedDict):
    Windows: str
    Linux: str
    output: str


@dc.dataclass
class CIInfo:
    url: str = ""
    cmake_var: str = ""
    uses_build_script: bool = False
    gdrive_url: GDriveURL | None = None
    tag: str = ""
    tag_prefix: str = ""

    def to_sdk_info(self) -> SDKInfo:
        return SDKInfo(url=self.url, cmake_var=self.cmake_var, tag=self.tag)


DEPENDENCIES: dict[str, CIInfo] = {
    "zeus": CIInfo(
        url="https://github.com/marovira/zeus.git",
        cmake_var="RAD_ZEUS_VERSION",
        uses_build_script=True,
    ),
    "opencv": CIInfo(
        url="",
        gdrive_url={
            "Windows": "https://drive.google.com/uc?id=1Dm7zbKo9yqyJY3qW_yZSrX7GgcOKA-2z",
            "Linux": "",
            "output": "opencv.zip",
        },
    ),
    "tbb": CIInfo(
        url="",
        gdrive_url={
            "Windows": "https://drive.google.com/uc?id=1uBVLn-8Ny2REmsO5nvnJBXJbTBR0r5QV",
            "Linux": "",
            "output": "TBB.zip",
        },
    ),
    "onnxruntime": CIInfo(
        url="",
        gdrive_url={
            "Windows": "https://drive.google.com/uc?id=1FdjUi9VaXdNjQOErClBF15n-521k8MM2",
            "Linux": "",
            "output": "onnxruntime.zip",
        },
    ),
}


def get_from_gdrive(name: str, info: CIInfo, install_root: pathlib.Path) -> None:
    assert info.gdrive_url is not None
    out_path = install_root / info.gdrive_url["output"]
    if not gdown.cached_download(
        info.gdrive_url[platform.system()],  # type: ignore[literal-required]
        str(out_path),
        postprocess=gdown.extractall,
    ):
        raise RuntimeError(f"error: unable to download archive for {name}")

    # Ensure we remove the archive
    out_path.unlink()


def check_dependencies(deps_root: pathlib.Path) -> bool:
    installed_deps = [folder.stem for folder in deps_root.iterdir() if folder.is_dir()]
    return all(name in installed_deps for name in DEPENDENCIES)


def install_dependencies(deps_root: pathlib.Path) -> None:
    if check_dependencies(deps_root):
        return

    existing_deps = [
        dir_name.stem for dir_name in deps_root.iterdir() if dir_name.is_dir()
    ]
    src_root = deps_root / "src"
    if src_root.exists():
        shutil.rmtree(src_root, onerror=rmtree_error)
    src_root.mkdir(exist_ok=True, parents=True)

    cur_dir = pathlib.Path.cwd()
    os.chdir(src_root)

    for name, info in DEPENDENCIES.items():
        if name in existing_deps:
            continue

        if info.gdrive_url is not None:
            get_from_gdrive(name, info, deps_root)
            continue

        clone_repo(name, info.to_sdk_info())
        if info.uses_build_script:
            build_from_script(name, src_root, deps_root / name, "Release")

            # Move all the installed folders to the deps root.
            for path in (deps_root / name).iterdir():
                if not path.is_dir():
                    continue
                path.rename(deps_root / path.stem)

            (deps_root / name).rmdir()
            continue

    os.chdir(cur_dir)
    shutil.rmtree(src_root, onerror=rmtree_error)


def read_versions(root: pathlib.Path) -> None:
    deps_file = root / "dependencies.txt"

    with deps_file.open("r", encoding="utf-8") as infile:
        lines = infile.read().splitlines()

    for line in lines:
        if "#" in line:
            continue

        elems = line.split("==")
        for sdk in DEPENDENCIES.values():
            if sdk.cmake_var == elems[0]:
                sdk.tag = sdk.tag_prefix + elems[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated dependency build script for RAD for CI"
    )
    parser.add_argument(
        "root",
        metavar="ROOT",
        type=str,
        nargs=1,
        help="The root where the dependencies will be built",
    )

    args = parser.parse_args()
    deps_root = pathlib.Path(args.root[0]).resolve()

    project_root = pathlib.Path(__file__).parent.parent

    read_versions(project_root)
    deps_root.mkdir(exist_ok=True, parents=True)
    install_dependencies(deps_root)


if __name__ == "__main__":
    main()
