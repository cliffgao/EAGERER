# EAGERER-docker
This is the docker version of the [EAGERER]:Precise estimation of residue relative solvent accessible area from Cα atom distance matrix using a deep learning method

You only need docker, access to dockerhub, and the files in this repository to run the tool. It is tested on Ubuntu 18.04 LTS (and docker version detailed in `docker_version.txt`), but it should work on any docker implementation. 

If you prefer to install EAGERER directly on your machine or you want to retrain the models, you should use the [repository](https://github.com/cliffgao/EAGERER), which requires you to set up the dependencies like torch . However for using the tool for prediction, it is strongly recommended to use the docker approach explained here. 

## Instructions for running EAGERER
after cloning (or downloading and extracting) the repository, the next step is to build the image by entering the following command in the root folder of the repository:
(including `.`)
```
docker image build -t eagerer:v1 .
```

This creates a docker image named `eagerer:v1`;then creates container through the image:
```
docker container run -p 80:6080 -it eagerer:v1 /bin/bash
```


## After docker successfully runs the image, root@d25dd6ce7ede:/EAGERER# is displayed. Enter "ls" on the command line, and EAGERER data and code will be found in the path.
## Usage


There are two ways to run EAGERER:

1. 
```
root@d25dd6ce7ede:/EAGERER# bash run.sh 
```
2. or  run *py

```
root@d25dd6ce7ede:/EAGERER# python ./bin/EAGERER.py  eg.list
```
## if you meet this error like "sh: 1: ./bin/test.py: Permission denied", you can enter the command "chmod 777 ./bin/test.py", then following the running steps.

## Example

```
root@d25dd6ce7ede:/EAGERER# sh run.sh 
```

### Input format:
EAGERER accepts one or more proteins saved in file-- 'eg.list'as input. The input should be proteinID or proteinID.pdb .
```
./yourpdbs/1a1xA
./yourpdbs/T1024.pdb
```

you can also look at the example file `eg.list`

### Output
the output is produced in file `yourpdbs`, you can find an example output included in the repository .


