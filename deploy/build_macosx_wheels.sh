#! /bin/bash
set -e
export package_version="${1:-$(date +'%Y.%m.%d.%H.%M.%S')}"
echo "Building macosx wheels, version '${package_version}'"

wheelhouse="${HOME}/Research/Temp/wheelhouse"

/bin/rm -rf "${wheelhouse}"
mkdir -p "${wheelhouse}"

CONDA_ENVS=( py27 py34 py35 py36 )

# Update conda envs
for CONDA_ENV in "${CONDA_ENVS[@]}"; do
    source activate "${CONDA_ENV}"
    conda update -y --all
    pip install --upgrade pip
    source deactivate
done

# Compile wheels
for CONDA_ENV in "${CONDA_ENVS[@]}"; do
    source activate "${CONDA_ENV}"
    pip install -r ./requirements.txt
    pip wheel ./ -w "${wheelhouse}/"
    source deactivate
done

# Bundle external shared libraries into the wheels
for whl in $(ls $(echo "${wheelhouse}/*.whl")); do
    echo
    delocate-listdeps --depending "$whl"
    delocate-wheel -v "$whl"
    delocate-listdeps --depending "$whl"
    echo
done


### NOTE: These lines are specialized for quaternion
for CONDA_ENV in "${CONDA_ENVS[@]}"; do
    source activate "${CONDA_ENV}"
    # Install packages and test ability to import and run simple command
    pip install --upgrade scri --no-index -f "${wheelhouse}"
    (cd "$HOME"; python -c 'import scri; print("scri version:", scri.__version__)')
    source deactivate
done


# Upload to pypi
echo "Uploading to pypi"
# pip install twine
# twine upload "${wheelhouse}"/*macosx*.whl
"${LAST_PYBIN}"/pip install twine
"${LAST_PYBIN}"/twine upload /wheelhouse/*macosx*.whl
