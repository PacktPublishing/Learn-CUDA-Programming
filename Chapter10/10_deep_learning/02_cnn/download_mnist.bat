echo off
REM setting variables
set zipPath="C:\Program Files\7-Zip\7z.exe"
set train_images="train-images-idx3-ubyte.gz"
set train_labels="train-labels-idx1-ubyte.gz"
set test_images="t10k-images-idx3-ubyte.gz"
set test_labels="t10k-labels-idx1-ubyte.gz"
set url_base="http://yann.lecun.com/exdb/mnist"

REM check if 7-zip installed
IF NOT EXIST %zipPath% GOTO NO_7ZIP

REM create dataset folder for the datasets
mkdir dataset
cd dataset

REM download datasets
curl -O %url_base%/%train_images%
%zipPath% e .\train-images-idx3-ubyte.gz
curl -O %url_base%/train-labels-idx1-ubyte.gz
%zipPath% e .\train-labels-idx1-ubyte.gz
curl -O %url_base%/t10k-images-idx3-ubyte.gz
%zipPath% e .\t10k-images-idx3-ubyte.gz
curl -O %url_base%/t10k-labels-idx1-ubyte.gz
%zipPath% e .\t10k-labels-idx1-ubyte.gz

exit

REM exception: no 7-zip found
:NO_7ZIP
echo "Please install 7-zip to extract downloaded MNIST dataset"
exit /b 1