FROM pytorchlightning/pytorch_lightning:base-cuda-py3.6-torch1.4

# Set up environment and renderer user
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install useful commands
RUN apt-get update && apt-get install -y \
      software-properties-common \
      cmake \
      git \
      curl wget \
      ca-certificates \
      nano \
      vim \
      htop

WORKDIR /home/root

# Start running
USER root
WORKDIR /home/root/contra

# Configure ssh
EXPOSE 22
RUN apt-get update
RUN apt-get install -y openssh-server gedit gnome-terminal tmux

RUN sed -i 's/PermitRootLogin/#PermitRootLogin/g' /etc/ssh/sshd_config
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN echo 'root:Aa123456' | chpasswd

# Init environment
RUN pip install pandas ipython jupyter pytorch-lightning optuna flask bokeh panel pathos tensorboard tensorflow datetime wandb sentencepiece==0.1.91 transformers scikit-learn

ENTRYPOINT ["/bin/bash"]
CMD []
