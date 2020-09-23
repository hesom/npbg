call conda create -y -n npbg python=3.6

call conda activate npbg
call conda install -y numpy pyyaml
call conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch

call pip install tensorboardX

call conda install -y munch scipy matplotlib Cython PyOpenGL pillow tqdm scikit-learn

call pip install PyOpenGL_accelerate trimesh huepy

call pip install git+https://github.com/DmitryUlyanov/glumpy numpy-quaternion

call conda install -y opencv

call git clone https://github.com/inducer/pycuda
cd pycuda
call git submodule update --init
call python configure.py --cuda-enable-gl
call python setup.py install
cd ..

call git clone git@github.com:glfw/glfw.git
cd glfw
mkdir build
cd build
call cmake -DBUILD_SHARED_LIBS=ON ..
call cmake --build . --target install
cd ../..

ECHO "Next steps:"
ECHO "Add glfw binary dir to PATH environment variable (most likely in C:\Program Files (x86)\GLFW\bin)"
ECHO "Close and reopen CMD (no need for admin privileges anymore)"
ECHO "Test with fitted scenes (see github repo)"