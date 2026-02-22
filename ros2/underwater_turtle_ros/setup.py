from setuptools import setup

package_name = "underwater_turtle_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/pn.launch.py"]),
        ("share/" + package_name + "/config", ["config/pn_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="M A Hafiz",
    maintainer_email="you@example.com",
    description="ROS2 node wrapper for underwater turtle PN pursuit controller",
    license="MIT",
    entry_points={
        "console_scripts": [
            "pn_node = underwater_turtle_ros.pn_node:main",
        ],
    },
)
