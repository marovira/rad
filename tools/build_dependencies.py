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

import gdown


class GDriveURL(typing.TypedDict):
    Windows: str
    Linux: str
    output: str


@dc.dataclass
class SDKInfo:
    url: str = ""
    tag: str = ""
    tag_prefix: str = ""
    flags: list[str] = dc.field(default_factory=list)
    cmake_var: str = ""
    is_extra_dependency: bool = False
    clone_recursive: bool = False
    archive_name: str | None = None
    gdrive_url: GDriveURL | None = None


DEPENDENCIES: dict[str, SDKInfo] = {
    "magic_enum": SDKInfo(
        url="https://github.com/Neargye/magic_enum.git",
        tag_prefix="v",
        flags=[
            "-DMAGIC_ENUM_OPT_BUILD_EXAMPLES=OFF",
            "-DMAGIC_ENUM_OPT_INSTALL=ON",
            "-DMAGIC_ENUM_OPT_BUILD_TESTS=OFF",
        ],
        cmake_var="ZEUS_MAGIC_ENUM_VERSION",
    ),
    "fmt": SDKInfo(
        url="https://github.com/fmtlib/fmt.git",
        flags=[
            "-DFMT_INSTALL=ON",
            "-DFMT_TEST=OFF",
        ],
        cmake_var="ZEUS_FMT_VERSION",
    ),
    "Catch2": SDKInfo(
        url="https://github.com/catchorg/Catch2.git",
        tag_prefix="v",
        flags=[
            "-DCATCH_INSTALL_DOCS=OFF",
        ],
        cmake_var="RAD_CATCH_VERSION",
    ),
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
        gdrive_url={
            "Windows": "https://drive.google.com/uc?id=1uBVLn-8Ny2REmsO5nvnJBXJbTBR0r5QV",
            "Linux": "https://drive.google.com/uc?id=1wdc5Emwa8Ft3a0fQRttLygs3UUF6wWw8",
            "output": "TBB.zip",
        },
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
        gdrive_url={
            "Windows": "https://drive.google.com/uc?id=1Dm7zbKo9yqyJY3qW_yZSrX7GgcOKA-2z",
            "Linux": "https://drive.google.com/uc?id=1whhqtS6_YSlcnF2_MzRFu0ZnvKa-T7Mn",
            "output": "opencv.zip",
        },
    ),
    "opencv_contrib": SDKInfo(
        url="https://github.com/opencv/opencv_contrib.git",
        cmake_var="RAD_OPENCV_VERSION",
        is_extra_dependency=True,
    ),
    "onnxruntime": SDKInfo(
        url="https://github.com/microsoft/onnxruntime.git",
        tag_prefix="v",
        flags=[
            "--build_shared_lib",
            "--parallel",
            "--skip_submodule_sync",
            "--use_dml" if platform.system() == "Windows" else "",
            "--skip_tests",
        ],
        cmake_var="RAD_ONNX_VERSION",
        clone_recursive=True,
        archive_name="onnxruntime.zip",
        gdrive_url={
            "Windows": "https://drive.google.com/uc?id=1FdjUi9VaXdNjQOErClBF15n-521k8MM2",
            "Linux": "https://drive.google.com/uc?id=1we8HonPhBK2AKsAsHOdmoCQRRFluX8cp",
            "output": "onnxruntime.zip",
        },
    ),
    "zeus": SDKInfo(
        url="https://github.com/marovira/zeus.git",
        cmake_var="RAD_ZEUS_VERSION",
        flags=[
            "-DZEUS_INSTALL_TARGET=ON",
        ],
    ),
}

BASE_DEPENDENCIES: list[str] = ["zeus"]


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
    args.append(f"-DCMAKE_PREFIX_PATH={install_root.parent}")

    execute_command(f"Configuring {name}", args)


def configure_opencv(
    name: str,
    sdk_info: SDKInfo,
    cmake_cfg: CMakeConfig,
    build_root: pathlib.Path,
    install_root: pathlib.Path,
    create_archives: bool = False,
) -> None:
    if not create_archives:
        clone_repo("opencv_contrib", DEPENDENCIES["opencv_contrib"])

        module_path = pathlib.Path().cwd() / "opencv_contrib/modules"
        sdk_info.flags.append(f"-DOPENCV_EXTRA_MODULES_PATH={module_path}")

    configure_cmake(name, sdk_info, cmake_cfg, build_root, install_root)


def build(name: str, build_root: pathlib.Path, config: str) -> None:
    args = ["cmake", "--build", f"{str(build_root)}", "--parallel", "--config", config]
    execute_command(f"Building {name}", args)


def build_onnxruntime(
    name: str, info: SDKInfo, install_root: pathlib.Path, config: str
) -> None:
    ort_root = pathlib.Path.cwd() / name
    build_script_name = "build.bat" if platform.system() == "Windows" else "build.sh"

    cur_dir = pathlib.Path.cwd()
    os.chdir(ort_root)

    script_cmd = (
        build_script_name if platform.system() == "Windows" else f"./{build_script_name}"
    )

    args = [
        script_cmd,
        "--config",
        config,
        "--target",
        "install",
    ]
    args.extend(info.flags)
    args.extend(
        [
            "--cmake_extra_defines",
            f"CMAKE_INSTALL_PREFIX={str(install_root)}",
            "onnxruntime_BUILD_UNIT_TESTS=OFF",
        ]
    )

    args = list(filter(None, args))
    execute_command(f"Installing {name}", args)
    os.chdir(cur_dir)

    if platform.system() == "Windows":
        build_root = ort_root / "build"
        dml_path = build_root / f"Windows/{config}/{config}/DirectML.dll"
        shutil.copy2(str(dml_path), f"{install_root / 'bin'}")


def install(name: str, build_root: pathlib.Path, config: str) -> None:
    args = ["cmake", "--install", f"{str(build_root)}", "--config", config]
    execute_command(f"Installing {name}", args)


def get_from_gdrive(name: str, info: SDKInfo, install_root: pathlib.Path) -> None:
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
    return all(
        name in installed_deps
        for name in DEPENDENCIES
        if not DEPENDENCIES[name].is_extra_dependency
    )


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


def set_sdk_versions(project_root: pathlib.Path, tmp_root: pathlib.Path) -> None:
    # Read the version numbers from the current project first.
    read_versions(project_root)

    # Now loop through the base external dependencies and fill in the versions for the
    # remaining SDKs.
    tmp_root.mkdir(exist_ok=True)
    cur_dir = pathlib.Path().cwd()
    os.chdir(tmp_root)

    for name in BASE_DEPENDENCIES:
        info = DEPENDENCIES[name]

        clone_repo(name, info)
        read_versions(tmp_root / name)

    os.chdir(cur_dir)
    shutil.rmtree(tmp_root, onerror=rmtree_error)


def generate_archives(deps_root: pathlib.Path, create_archives: bool) -> None:
    if not create_archives:
        return

    prog = "7z" if platform.system() == "Windows" else "zip"
    if shutil.which(prog) is None:
        raise RuntimeError(f"error: {prog} is required to generate archives")

    cur_dir = pathlib.Path.cwd()
    os.chdir(deps_root)

    for name, info in DEPENDENCIES.items():
        if info.archive_name is None:
            continue

        args = [prog]
        if platform.system() == "Windows":
            args.append("a")

        args.extend([info.archive_name, "-r", name])
        execute_command(f"Archiving {name}", args)

    os.chdir(cur_dir)


def install_dependencies(
    deps_root: pathlib.Path,
    config: str,
    create_archives: bool = False,
    ci_build: bool = False,
) -> None:
    if check_dependencies(deps_root):
        generate_archives(deps_root, create_archives)
        return

    project_root = pathlib.Path(__file__).parent.parent
    existing_deps = [
        dir_name.stem for dir_name in deps_root.iterdir() if dir_name.is_dir()
    ]
    src_root = deps_root / "src"
    if src_root.exists():
        shutil.rmtree(src_root, onerror=rmtree_error)
    src_root.mkdir(exist_ok=True, parents=True)

    cur_dir = pathlib.Path.cwd()
    os.chdir(src_root)

    set_sdk_versions(project_root, src_root / "tmp")

    cmake_cfg = get_cmake_config(project_root)

    for name, info in DEPENDENCIES.items():
        install_root = deps_root / name
        build_root = src_root / (name + "/build")

        if name in existing_deps:
            continue

        if info.is_extra_dependency:
            continue

        if create_archives and info.archive_name is None:
            continue

        if ci_build and info.gdrive_url is not None:
            get_from_gdrive(name, info, deps_root)
            continue

        clone_repo(name, info)
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
                create_archives=create_archives,
            )
        else:
            configure_cmake(name, info, cmake_cfg, build_root, install_root)
        build(name, build_root, config)
        install(name, build_root, config)

    os.chdir(cur_dir)
    shutil.rmtree(src_root, onerror=rmtree_error)
    generate_archives(deps_root, create_archives)


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
    parser.add_argument("-c", "--ci", action="store_true", help="Build for CI")

    args = parser.parse_args()

    deps_root = pathlib.Path(args.root[0]).resolve()
    config = args.build[0] if args.build is not None else "Release"

    # Only generate release builds for CI, since we never test Debug.
    if args.archive:
        config = "Release"

    deps_root.mkdir(exist_ok=True, parents=True)
    install_dependencies(deps_root, config, args.archive, args.ci)


if __name__ == "__main__":
    main()
