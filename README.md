# Contra
Repository for the Contra project


run:
```
sudo docker build -t contra:1.0 .
```
to build the  docker images. Then run:
```
sudo docker run -it -v /home/urielsinger/contra:/root/contra -v /home/urielsinger/common:/root/common -p 1234:22 -p 8000-8020:8000-8020 --gpus all --name contra contra:1.0
```
to get a container for the project.

### run tensorboard
```
tensorboard --logdir /root/contra/logdir --host=0.0.0.0 --port=8000
```
### run jupyter
```
jupyter notebook --no-browser --ip=0.0.0.0 --port=8001 --allow-root
```