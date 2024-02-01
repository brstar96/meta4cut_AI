FROM nvidia/cuda:11.3.0-base-ubuntu20.04
MAINTAINER MYEONGGYULEE <brstar96@naver.com>

# set ubuntu dependencies
WORKDIR /home
RUN mkdir /root/.ssh
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# set host dependencies
RUN apt-get update -y && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update \
 && apt-get install -y wget curl git unzip vim screen \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# set miniconda
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
	/bin/bash ~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh && \
	ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
	echo "conda activate base" >> ~/.bashrc

# fetch environment setting repository & setup envirinments
RUN conda env create -f environment_meta4cut.yml
RUN echo "conda activate meta4cut" > ~/.bashrc
RUN pip install -r requirements.txt
# install dlib library
RUN apt-get update && apt-get install -y build-essential cmake
RUN apt-get install -y libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
RUN git clone https://github.com/davisking/dlib.git
WORKDIR dlib
RUN mkdir build; cd build; cmake ..; cmake --build .
WORKDIR ..
RUN python3 setup.py install
# install mmcv & mmdet
RUN pip install -U openmim
RUN mim install mmcv-full
RUN mim install mmdet

# setup openssh
RUN apt-get update
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd

# set password
RUN echo 'root:meta4cut' |chpasswd

#replace sshd_config
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^#?PubkeyAuthentication\s+.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^#?StirictModes\s+.*/StirictModes yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^#?PasswordAuthentication\s+.*/PasswordAuthentication no/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN echo 'RSAAuthentication yes' >> /etc/ssh/sshd_config
RUN echo '/usr/sbin/sshd' >> ~/.bashrc
RUN service ssh start

WORKDIR /home
# ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]