import argparse
import dataclasses as dc
import json
import os
import pathlib
import platform
import shutil
import stat
import subprocess
import typing


@dc.dataclass
class SDKInfo:
    url: str = ""
    tag: str = ""
    tag_prefix: str = ""
    flags: list[str] = dc.field(default_factory=list)
    cmake_var: str = ""
    is_extra_dependency: bool = False
    uses_build_script: bool = False
    clone_recursive: bool = False
    archive_name: str | None = None


DEPENDENCIES: dict[str, SDKInfo] = {
    "tbb": SDKInfo(
        url="https://github.com/oneapi-src/oneTBB.git",
        tag_prefix="v",
        flags=[
            "-DTBB_TEST=OFF",
            "-DTBB_EXAMPLES=OFF",
            "-DTBB_INSTALL=ON",
        ],
        cmake_var="RAD_TBB_VERSION",
        archive_name="TBB.zip",
    ),
    "opencv": SDKInfo(
        url="https://github.com/opencv/opencv.git",
        flags=[
            "-DWITH_TBB=ON",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_opencv_world=ON",
            "-DBUILD_opencv_python3=OFF",
            "-DBUILD_opencv_python_bindings_generator=OFF",
            "-DBUILD_PERF_TESTS=OFF",
            "-DBUILD_TESTS=OFF",
            "-DWITH_DIRECTX=OFF",
        ],
        cmake_var="RAD_OPENCV_VERSION",
        archive_name="opencv.zip",
    ),
    "opencv_contrib": SDKInfo(
        url="https://github.com/opencv/opencv_contrib.git",
        cmake_var="RAD_OPENCV_VERSION",
        is_extra_dependency=True,
    ),
    "onnxruntime": SDKInfo(
        url="https://github.com/microsoft/onnxruntime.git",
        tag_prefix="v",
        cmake_var="RAD_ONNX_VERSION",
        clone_recursive=True,
        archive_name="onnxruntime.zip",
    ),
    "zeus": SDKInfo(
        url="https://github.com/marovira/zeus.git",
        uses_build_script=True,
        cmake_var="RAD_ZEUS_VERSION",
    ),
}


@dc.dataclass
class CMakeConfig:
    generator: str = ""
    cxx_standard: str = ""

    def get_generator_flags(self) -> list[str]:
        return ["-G", self.generator]

    def get_cxx_standard_flag(self) -> str:
        return f"-DCMAKE_CXX_STANDARD={self.cxx_standard}"


def rmtree_error(func, path, exc_info):
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)  # noqa: PTH101
        func(path)
    else:
        raise exc_info


def find_preset(presets: list[dict[str, typing.Any]], name: str) -> dict[str, typing.Any]:
    for preset in presets:
        if preset["name"] == name:
            return preset

    raise KeyError(f"error: CMakePresets file does not contain a preset called {name}")


def get_cmake_config(project_root: pathlib.Path) -> CMakeConfig:
    preset_path = project_root / "CMakePresets.json"
    with preset_path.open("rb") as infile:
        preset_data = json.load(infile)

        presets = preset_data["configurePresets"]
        config = CMakeConfig()
        if platform.system() == "Windows":
            base_preset = find_preset(presets, "msvc")
        else:
            base_preset = find_preset(presets, "unix_base")

        config.generator = base_preset["generator"]
        config.cxx_standard = base_preset["cacheVariables"]["CMAKE_CXX_STANDARD"]

        return config


def execute_command(proc_name: str, args: list[str]) -> None:
    try:
        env = os.environ.copy()
        subprocess.run(args, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{proc_name} finised with a non-zero exit code") from exc


def clone_repo(name: str, info: SDKInfo) -> None:
    args = ["git", "clone", info.url, name, "--branch", info.tag, "--depth", "1"]
    if info.clone_recursive:
        args.append("--recurse-submodules")

    execute_command(f"Cloning {name}", args)


def configure_cmake(
    name: str,
    sdk_info: SDKInfo,
    cmake_cfg: CMakeConfig,
    build_root: pathlib.Path,
    install_root: pathlib.Path,
) -> None:
    args = ["cmake"]
    args.extend(cmake_cfg.get_generator_flags())
    args.extend(
        ["-S", name, "-B", f"{str(build_root)}", cmake_cfg.get_cxx_standard_flag()]
    )
    args.extend(sdk_info.flags)
    args.append(f"-DCMAKE_INSTALL_PREFIX={str(install_root)}")
    args.append("-DCMAKE_DEBUG_POSTFIX=d")

    if sdk_info.uses_build_script:
        args.append(f"-DCMAKE_PREFIX_PATH={install_root.parent}")

    execute_command(f"Configuring {name}", args)


def configure_opencv(
    name: str,
    sdk_info: SDKInfo,
    cmake_cfg: CMakeConfig,
    build_root: pathlib.Path,
    install_root: pathlib.Path,
    generating_archive: bool = False,
) -> None:
    assert name == "opencv"

    if not generating_archive:
        clone_repo("opencv_contrib", DEPENDENCIES["opencv_contrib"])

        module_path = pathlib.Path().cwd() / "opencv_contrib/modules"
        sdk_info.flags.append(f"-DOPENCV_EXTRA_MODULES_PATH={module_path}")

    configure_cmake(name, sdk_info, cmake_cfg, build_root, install_root)


def build(name: str, build_root: pathlib.Path, config: str | None) -> None:
    assert config in (None, "Debug", "Release")

    args = ["cmake", "--build", f"{str(build_root)}", "--parallel"]
    if config is not None:
        args.extend(["--config", config])
    execute_command(f"Building {name}", args)


def build_from_script(
    name: str,
    src_root: pathlib.Path,
    install_root: pathlib.Path,
    config: str | None,
) -> None:
    assert config in (None, "Debug", "Release")

    root = src_root / name

    args = [
        "python" if platform.system() == "Windows" else "python3",
        f"{root / 'tools/build_dependencies.py'}",
        str(install_root),
    ]

    if config is not None:
        args.extend(["-b", config])

    execute_command(f"Building {name}", args)


def build_onnxruntime(
    name: str, info: SDKInfo, install_root: pathlib.Path, config: str | None
) -> None:
    assert name == "onnxruntime"
    assert config in (None, "Debug", "Release")

    ort_root = pathlib.Path.cwd() / name
    build_script_name = "build.bat" if platform.system() == "Windows" else "build.sh"

    cur_dir = pathlib.Path.cwd()
    os.chdir(ort_root)

    script_cmd = (
        build_script_name if platform.system() == "Windows" else f"./{build_script_name}"
    )

    if config is None:
        config = "Release"

    args = [
        script_cmd,
        "--config",
        config,
        "--target",
        "install",
        "--build_shared_lib",
        "--parallel",
        "--skip_submodule_sync",
        "--use_dml" if platform.system() == "Windows" else "",
        "--skip_tests",
        "--cmake_extra_defines",
        f"CMAKE_INSTALL_PREFIX={str(install_root)}",
    ]
    args = list(filter(None, args))
    execute_command(f"Installing {name}", args)
    os.chdir(cur_dir)


def install(name: str, build_root: pathlib.Path, config: str | None) -> None:
    assert config in (None, "Debug", "Release")

    args = ["cmake", "--install", f"{str(build_root)}"]
    if config is not None:
        args.extend(["--config", config])
    execute_command(f"Installing {name}", args)


def check_dependencies(deps_root: pathlib.Path) -> bool:
    installed_deps = [folder.stem for folder in deps_root.iterdir() if folder.is_dir()]
    return all(
        name in installed_deps
        for name in DEPENDENCIES
        if not DEPENDENCIES[name].is_extra_dependency
    )


def compress_folder(name: str, root: pathlib.Path, info: SDKInfo) -> None:
    assert info.archive_name is not None

    prog = "7z" if platform.system() == "Windows" else "zip"
    if shutil.which(prog) is None:
        raise RuntimeError(f"error: {prog} is required to generate archives")

    args = [prog]
    if platform.system() == "Windows":
        args.append("a")

    install_root = root.parent
    args.extend(
        [
            info.archive_name,
            "-r",
            str(root)
            if platform.system() == "Windows"
            else str(root.relative_to(install_root)),
        ]
    )

    cur_dir = pathlib.Path().cwd()
    if platform.system() == "Linux":
        os.chdir(install_root)

    execute_command(f"Archiving {name}", args)
    os.chdir(cur_dir)


def install_dependencies(
    cmake_cfg: CMakeConfig,
    deps_root: pathlib.Path,
    config: str | None,
    generate_archive: bool = False,
) -> None:
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

        if info.is_extra_dependency:
            continue

        if generate_archive and info.archive_name is None:
            continue

        install_root = deps_root / name
        build_root = src_root / (name + "/build")

        clone_repo(name, info)
        if info.uses_build_script:
            build_from_script(name, src_root, install_root, config)

            # Move all the installed folders to the deps root.
            for path in (deps_root / name).iterdir():
                if not path.is_dir():
                    continue
                path.rename(deps_root / path.stem)

            (deps_root / name).rmdir()

        if name == "onnxruntime":
            build_onnxruntime(name, info, install_root, config)
            continue

        if name == "opencv":
            configure_opencv(
                name,
                info,
                cmake_cfg,
                build_root,
                install_root,
                generating_archive=generate_archive,
            )
        else:
            configure_cmake(name, info, cmake_cfg, build_root, install_root)
        build(name, build_root, config)
        install(name, build_root, config)

    os.chdir(deps_root)
    shutil.rmtree(src_root, onerror=rmtree_error)

    if generate_archive:
        for path in deps_root.iterdir():
            if not path.is_dir():
                continue
            compress_folder(path.stem, path, DEPENDENCIES[path.stem])

    os.chdir(cur_dir)


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
        description="Automated dependency build script for RAD"
    )
    parser.add_argument(
        "root",
        metavar="ROOT",
        type=str,
        nargs=1,
        help="The root where the dependencies will be built",
    )
    parser.add_argument(
        "-b",
        "--build",
        metavar="BUILD",
        type=str,
        nargs=1,
        help="Build configuration to use. Must be one of 'Debug' or 'Release'",
    )
    parser.add_argument(
        "-a", "--archive", action="store_true", help="Generate archives for CI builds"
    )

    args = parser.parse_args()

    deps_root = pathlib.Path(args.root[0]).resolve()
    config = args.build[0] if args.build is not None else None

    # Only generate release builds for CI, since we never test Debug.
    if args.archive:
        config = "Release"

    project_root = pathlib.Path(__file__).parent.parent

    read_versions(project_root)
    cmake_cfg = get_cmake_config(project_root)

    deps_root.mkdir(exist_ok=True, parents=True)
    install_dependencies(cmake_cfg, deps_root, config, args.archive)


if __name__ == "__main__":
    main()
