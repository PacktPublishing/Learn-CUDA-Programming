Nsight Eclipse Edition requires java 1.7 for their java runtime engine.

However, the latest linux platform does not provides java 7 from their support so it is requred to install java 7 manually.
To tell the truth, JRE is installed with CUDA toolkit. However it installs the latest one and does not resolve the dependency issue. 


Firstly, download JRE from the oracle's [site](https://www.oracle.com/technetwork/java/javase/downloads/java-archive-downloads-javase7-521261.html).

Untar the file and move the files into the proper path.
```bash
$ tar xzf jdk-7u80-linux-x64.tar.gz
$ sudo mkdir /usr/lib/jvm
$ sudo mv jdk1.7.0_80 /usr/lib/jvm
```

In general, the system will use the latest java version. To set to use older java version, select the older version with this command.

```bash
$ sudo update-alternatives --config java

```

For example, update-alternatives gives several installed java version.

```
There are 2 choices for the alternative java (providing /usr/bin/java).

  Selection    Path                                         Priority   Status
------------------------------------------------------------
* 0            /usr/lib/jvm/java-11-openjdk-amd64/bin/java   1111      auto mode
  1            /usr/lib/jvm/java-11-openjdk-amd64/bin/java   1111      manual mode
  2            /usr/lib/jvm/jre1.7.0_80/bin/java             1         manual mode

Press <enter> to keep the current choice[*], or type selection number:
```


Put number 2 in this case to use java 1.7.0.

Do this for the rest of JRE runtime file.
```bash
$ sudo update-alternavies --config javaws
```
