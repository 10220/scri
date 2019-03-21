#! /bin/bash

if [ -z "$ANACONDA_API_TOKEN" ]; then
    echo "The variable 'ANACONDA_API_TOKEN' cannot be empty"
    exit 1
fi

# if [[ -z $(git status -uno -s > /dev/null) ]]; then
#     echo "All changes must be git committed"
#     exit 1
# fi

export package_version=$(git log -1 --format=%cd --date=format:'%Y.%m.%d.%H.%M.%S' || date +"%Y.%m.%d.%H.%M.%S")
echo "Building version '${package_version}'"

# Rebuild and install locally, then test trivial action, to ensure there are no warnings
/bin/rm -rf build __pycache__
CFLAGS='-Werror -Wall -Wextra' python setup.py install
python -c 'import scri; print(scri.__version__)'

# Create a pure source pip package
/bin/rm -rf dist
python setup.py sdist
twine upload dist/*

# # Create a pure source pip package
# python setup.py sdist upload

# Create all the osx binary pip packages
./deploy/build_macosx_wheels.sh "${package_version}"

# Create all the osx conda packages
conda build .

# Start docker for the linux packages
open --hide --background -a Docker
while ! (docker ps > /dev/null 2>&1); do
    echo "Waiting for docker to start..."
    sleep 1
done

# Create all the linux binary pip packages on centos 5
docker run -i -t \
    -v ${HOME}/.pypirc:/root/.pypirc:ro \
    -v `pwd`:/code \
    -v `pwd`/deploy/build_manylinux_wheels.sh:/build_manylinux_wheels.sh \
    quay.io/pypa/manylinux1_x86_64 /build_manylinux_wheels.sh "${package_version}"

# Create all the linux binary conda packages on centos 6
docker run -i -t \
    -e package_version \
    -e ANACONDA_API_TOKEN \
    -v ${HOME}/.condarc:/root/.condarc:ro \
    -v `pwd`:/code \
    moble/miniconda-centos bash -c 'conda build .'
